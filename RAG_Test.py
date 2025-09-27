# -*- coding: utf-8 -*-
"""
股票新聞分析工具（完整 RAG 版 + 全文打分 + Ollama 分析 + Token 傾向顯示）
"""

import os
import signal
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime, timedelta, timezone
import regex as re
import numpy as np
import faiss
from google.cloud import firestore
from dotenv import load_dotenv
import time
from groq import Groq
from rich import print as rprint  # rich 專用

# ---------- 讀 .env ----------
def load_env():
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
        print(f"[info] 已載入 .env：{os.path.abspath('.env')}")
    else:
        load_dotenv(override=True)
        print("[info] 未找到 .env，改用系統環境變數")
load_env()

# ---------- 參數 ----------
TOKENS_COLLECTION = os.getenv("FIREBASE_TOKENS_COLLECTION", "bull_tokens")
NEWS_COLLECTION   = os.getenv("FIREBASE_NEWS_COLLECTION", "NEWS")
SCORE_THRESHOLD   = float(os.getenv("SCORE_THRESHOLD", "3.0"))
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "3"))
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "mistral")
TITLE_LEN         = 20

TAIWAN_TZ = timezone(timedelta(hours=8))
DOCID_RE  = re.compile(r"^(?P<ymd>\d{8})_(?P<hms>\d{6})$")

# ---------- 中斷控制 ----------
STOP = False
def _sigint_handler(signum, frame):
    global STOP
    if not STOP:
        STOP = True
        print("\n[info] 偵測到 Ctrl+C，將儘快安全停止…")
    else:
        raise KeyboardInterrupt
signal.signal(signal.SIGINT, _sigint_handler)

# ---------- 小工具 ----------
def normalize(text: str) -> str:
    if not text: return ""
    return re.sub(r"\s+", " ", text.strip()).lower()

def shorten_text(text: str, max_len: int = 500) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text

def parse_docid_time(doc_id: str):
    m = DOCID_RE.match(doc_id)
    if not m: return None
    try:
        return datetime.strptime(m.group("ymd")+m.group("hms"), "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

# ---------- 結構 ----------
@dataclass
class Token:
    polarity: str
    ttype: str
    pattern: str
    weight: float
    note: str

@dataclass
class MatchResult:
    score: float
    hits: List[Tuple[str, float, str]]

# ---------- Firestore ----------
def get_db() -> firestore.Client:
    db = firestore.Client()
    print(f"[info] GCP 專案：{db.project}")
    return db

def load_tokens(db: firestore.Client, col_name: str) -> Tuple[List[Token], List[Token]]:
    pos, neg = [], []
    for d in db.collection(col_name).stream():
        data = d.to_dict() or {}
        polarity = (data.get("polarity") or "").lower()
        ttype    = (data.get("type") or "substr").lower()
        patt     = str(data.get("pattern") or "")
        note     = str(data.get("note") or "")
        try:
            weight = float(data.get("weight", 1.0))
        except:
            weight = 1.0
        if not patt or polarity not in ("positive","negative"): continue
        (pos if polarity=="positive" else neg).append(Token(polarity, ttype, patt, weight, note))
    return pos, neg

def load_news_items(db: firestore.Client, col_name: str, lookback_days: int) -> List[Dict]:
    items: List[Dict] = []
    seen = set()
    now_tw = datetime.now(TAIWAN_TZ)
    start_tw = now_tw - timedelta(days=lookback_days)
    for d in db.collection(col_name).stream():
        dt = parse_docid_time(d.id)
        if dt and dt < start_tw: 
            continue
        data = d.to_dict() or {}
        for key, val in data.items():
            if not (isinstance(key,str) and key.startswith("news_") and isinstance(val,dict)): 
                continue
            title   = str(val.get("title") or "").strip()
            content = str(val.get("content") or "").strip()
            if not title and not content: 
                continue
            uniq_key = f"{title}|{content}"
            if uniq_key in seen:
                continue
            seen.add(uniq_key)
            items.append({
                "id": f"{d.id}#{key}",
                "title": title,
                "content": content,
                "ts": dt
            })
    items.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=TAIWAN_TZ), reverse=True)
    return items

# ---------- Token 編譯 / 打分 ----------
def compile_tokens(tokens: List[Token]):
    out = []
    for t in tokens:
        w = t.weight
        if t.polarity == "negative": w = -abs(w)
        if t.ttype == "regex":
            try:
                cre = re.compile(t.pattern, flags=re.IGNORECASE)
                out.append(("regex", cre, w, t.note, t.pattern))
            except:
                continue
        else:
            out.append(("substr", None, w, t.note, t.pattern.lower()))
    return out

def score_text(text: str, pos_compiled, neg_compiled) -> MatchResult:
    norm = normalize(text)
    score = 0.0
    hits = []
    seen_patterns = set()
    for ttype, cre, w, note, patt in pos_compiled + neg_compiled:
        key = (ttype, patt.lower() if ttype=="substr" else patt)
        if key in seen_patterns:
            continue
        matched = False
        if ttype=="regex" and cre.search(norm):
            matched = True
        elif ttype=="substr" and patt in norm:
            matched = True
        if matched:
            score += w
            hits.append((patt, w, note))
            seen_patterns.add(key)
    return MatchResult(score, hits)

def print_token_hits(hits: List[Tuple[str,float,str]]):
    """修改後：命中 token 顯示 + 統計傾向"""
    pos_count, neg_count = 0, 0
    for patt, w, note in hits:
        why = f"（{note}）" if note else ""
        if w > 0:
            rprint(f"  [green]+ 「{patt}」[/green]{why}")
            pos_count += 1
        elif w < 0:
            rprint(f"  [red]- 「{patt}」[/red]{why}")
            neg_count += 1
        else:
            rprint(f"  「{patt}」{why}")
    tendency = "偏多" if pos_count > neg_count else "偏空" if neg_count > pos_count else "中性"
    rprint(f"整體傾向：[green]+{pos_count}[/green] / [red]-{neg_count}[/red] → {tendency}")

# ---------- Embedding / FAISS ----------
def embed_text(text: str):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(1536, dtype=np.float32)

def build_faiss_index(vectors: List[np.ndarray], news_items: List[Dict]):
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    vec_matrix = np.vstack(vectors)
    faiss.normalize_L2(vec_matrix)
    index.add(vec_matrix)
    return index, news_items

# ---------- Ollama 分析 ----------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def prepare_news_for_llm(news_items: List[str]) -> str:
    out_lines = []
    for i, txt in enumerate(news_items, 1):
        out_lines.append(f"新聞 {i}：\n{shorten_text(txt, 200)}\n")
    return "\n".join(out_lines)

def retrieve_similar_news(index, vectors, top_k=3):
    results = []
    for v in vectors:
        faiss.normalize_L2(v.reshape(1, -1))
        D, I = index.search(v.reshape(1, -1), top_k)
        results.append(I[0].tolist())
    return results

def ollama_analyze(texts: List[str], retries=3, delay=1.5) -> str:
    global STOP
    combined_text = prepare_news_for_llm(texts)

    prompt = f"""以下是多則與台積電相關的新聞摘要。
請嚴格按照以下格式輸出，且只能輸出一次，不要重複，也不要其他文字：

明天台積電股價走勢：<只能輸出 上漲 或 下跌 或 不明確>
原因：<100字以內，簡要原因>

新聞摘要：

{combined_text}
"""

    for i in range(retries):
        if STOP:
            return "[stopped] 使用者中斷"
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Groq 免費模型，速度快
                messages=[
                    {"role": "system", "content": "你是一個專業的股市新聞分析助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            raw_result = response.choices[0].message.content.strip()

            # 照舊處理輸出
            m1 = re.search(r"明天台積電股價走勢[:：]?\s*(上漲|下跌|不明確)", raw_result)
            m2 = re.search(r"原因[:：]?\s*([\s\S]{1,400}?)\s*(?:\n明天|\Z)", raw_result)
            concl = m1.group(1) if m1 else "不明確"
            reason = m2.group(1).strip() if m2 else ""
            reason = reason.replace("\n", " ").strip()
            if len(reason) > 100:
                reason = reason[:100] + "…"

            return f"明天台積電股價走勢：{concl}\n原因：{reason if reason else '未取得原因或格式不符'}"

        except Exception as e:
            print(f"[warn] Groq 呼叫失敗 ({i+1}/{retries})：{e}")
            time.sleep(delay)

    return "[error] Groq 呼叫失敗"


# ---------- 主程式 ----------
def main():
    global STOP
    print(f"[info] 設定：tokens='{TOKENS_COLLECTION}', news='{NEWS_COLLECTION}', threshold={SCORE_THRESHOLD}")

    db = get_db()
    pos, neg = load_tokens(db, TOKENS_COLLECTION)
    pos_c = compile_tokens(pos)
    neg_c = compile_tokens(neg)

    items = load_news_items(db, NEWS_COLLECTION, LOOKBACK_DAYS)
    if not items:
        print("[info] NEWS 無資料可分析。")
        return

    filtered: List[Tuple[Dict, MatchResult]] = []
    for item in items:
        if STOP: break
        text_for_scoring = item["content"] or item["title"]
        res = score_text(text_for_scoring, pos_c, neg_c)
        if abs(res.score) >= SCORE_THRESHOLD:
            filtered.append((item, res))

    print(f"[info] 過濾後新聞: {len(filtered)} / {len(items)}")
    if not filtered or STOP:
        print("[info] 無新聞達到閾值或已中斷，結束")
        return

    vectors = [embed_text(n[0]["content"] or n[0]["title"]) for n in filtered]
    index, news_list = build_faiss_index(vectors, [n[0] for n in filtered])

    for item, res in filtered:
        short_title = item["title"] if item["title"] else item["content"]
        print(f"\n[{item['id']}] {short_title}")
        if res.score >= SCORE_THRESHOLD:
            print("✅ 明日可能大漲")
        elif res.score <= -SCORE_THRESHOLD:
            print("⚠️ 明日可能大跌")
        else:
            print("—")
        if res.hits:
            print("命中：")
            print_token_hits(res.hits)

    similar_idxs = retrieve_similar_news(index, vectors, top_k=3)
    rag_texts = []
    for i, idxs in enumerate(similar_idxs):
        texts = [filtered[i][0]["content"] or filtered[i][0]["title"]]
        for j in idxs:
            if j != i and j < len(filtered):
                texts.append(filtered[j][0]["content"] or filtered[j][0]["title"])
        rag_texts.append("\n".join(texts))

    # Ollama 分析輸出純文字
    print("\n--- Ollama 生成分析 ---")
    summary = ollama_analyze(rag_texts)
    print(summary)

if __name__ == "__main__":
    main()
