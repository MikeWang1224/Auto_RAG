# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海）
改進：新聞命中只分析包含公司名稱的句子，避免交叉誤判。
"""

import os, signal, time, regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 讀 .env ----------
if os.path.exists(".env"):
    load_dotenv(".env", override=True)
    print(f"[info] 已載入 .env：{os.path.abspath('.env')}")
else:
    load_dotenv(override=True)
    print("[info] 未找到 .env，改用系統環境變數")

# ---------- 常數 ----------
TOKENS_COLLECTION = os.getenv("FIREBASE_TOKENS_COLLECTION", "bull_tokens")
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "3.0"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "3"))
TAIWAN_TZ = timezone(timedelta(hours=8))

COMPANY_KEYWORDS = {
    "台積電": ["台積電", "TSMC", "2330"],
    "鴻海": ["鴻海", "Foxconn", "2317"],
}

STOP = False
def _sigint_handler(signum, frame):
    global STOP
    STOP = True
    print("\n[info] 偵測到 Ctrl+C，將安全停止…")
signal.signal(signal.SIGINT, _sigint_handler)

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

# ---------- 工具 ----------
DOCID_RE = re.compile(r"^(?P<ymd>\d{8})_(?P<hms>\d{6})$")
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def shorten_text(t: str, n=200):
    return t[:n] + "…" if len(t) > n else t

def first_n_sentences(text: str, n: int = 3) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[。\.！!\?？；;])\s*', text.strip())
    parts = [p for p in parts if p.strip()]
    if not parts:
        return text.strip()
    selected = parts[:n]
    joined = "".join(selected)
    if not re.search(r'[。\.！!\?？；;]\s*$', joined):
        joined = joined + "..."
    return joined

def parse_docid_time(doc_id: str):
    m = DOCID_RE.match(doc_id)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ymd")+m.group("hms"), "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

def extract_company_related_text(text: str, company: str) -> str:
    """
    從文章中取出包含該公司關鍵字的句子。
    若沒找到，回傳空字串（代表該新聞與公司無關）。
    """
    keywords = COMPANY_KEYWORDS.get(company, [company])
    sentences = re.split(r'[。！？；.!?]', text)
    related = [s.strip() for s in sentences if any(kw in s for kw in keywords)]
    return "。".join(related)

# ---------- Firestore ----------
def get_db():
    return firestore.Client()

def load_tokens(db, col) -> Tuple[List[Token], List[Token]]:
    pos, neg = [], []
    for d in db.collection(col).stream():
        data = d.to_dict() or {}
        pol = (data.get("polarity") or "").lower()
        ttype = (data.get("type") or "substr").lower()
        patt = str(data.get("pattern") or "")
        note = str(data.get("note") or "")
        try:
            w = float(data.get("weight", 1.0))
        except:
            w = 1.0
        if not patt or pol not in ("positive","negative"):
            continue
        (pos if pol=="positive" else neg).append(Token(pol, ttype, patt, w, note))
    return pos, neg

def load_news_items(db, col_name: str, days: int) -> List[Dict]:
    items, seen = [], set()
    now = datetime.now(TAIWAN_TZ)
    start = now - timedelta(days=days)
    for d in db.collection(col_name).stream():
        dt = parse_docid_time(d.id)
        if dt and dt < start:
            continue
        data = d.to_dict() or {}
        for k, v in data.items():
            if not (k.startswith("news_") and isinstance(v, dict)):
                continue
            title = str(v.get("title") or "")
            content = str(v.get("content") or "")
            if not title and not content:
                continue
            uniq = f"{title}|{content}"
            if uniq in seen:
                continue
            seen.add(uniq)
            items.append({"id": f"{d.id}#{k}", "title": title, "content": content, "ts": dt})
    items.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=TAIWAN_TZ), reverse=True)
    return items

# ---------- Token 打分 ----------
def compile_tokens(tokens: List[Token]):
    out = []
    for t in tokens:
        w = t.weight if t.polarity == "positive" else -abs(t.weight)
        if t.ttype == "regex":
            try:
                cre = re.compile(t.pattern, flags=re.IGNORECASE)
                out.append(("regex", cre, w, t.note, t.pattern))
            except:
                continue
        else:
            out.append(("substr", None, w, t.note, t.pattern.lower()))
    return out

def score_text(text: str, pos_c, neg_c) -> MatchResult:
    norm = normalize(text)
    score, hits, seen = 0.0, [], set()
    for ttype, cre, w, note, patt in pos_c + neg_c:
        key = (ttype, patt)
        if key in seen:
            continue
        matched = cre.search(norm) if ttype == "regex" else patt in norm
        if matched:
            score += w
            hits.append((patt, w, note))
            seen.add(key)
    return MatchResult(score, hits)

# ---------- Groq ----------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def prepare_news_for_llm(news_items: List[str]) -> str:
    return "\n".join(f"新聞 {i}：\n{shorten_text(t)}\n" for i,t in enumerate(news_items,1))

def ollama_analyze(texts: List[str], target: str) -> str:
    combined = prepare_news_for_llm(texts)
    prompt = f"""你是一位台灣股市研究員。根據以下新聞，判斷「明天{target}股價」最可能走勢。
請只回覆以下兩行格式（不要多餘文字）：

明天{target}股價走勢：<上漲 / 下跌 / 不明確>
原因：<40字以內，一句話簡潔說明主要理由>

新聞摘要：
{combined}
"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是專業股市新聞分析員，回答簡潔準確。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        cleaned = re.sub(r"^```(?:\w+)?|```$", "", raw).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned
    except Exception as e:
        return f"[error] Groq 呼叫失敗：{e}"

# ---------- 分析 ----------
def analyze_target(db, news_col: str, target: str, result_col: str):
    print(f"\n[info] ===== 開始分析 {target} =====")
    pos, neg = load_tokens(db, TOKENS_COLLECTION)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)

    items = load_news_items(db, news_col, LOOKBACK_DAYS)
    if not items:
        print(f"[info] {news_col} 無資料")
        return

    filtered, local_log = [], []
    terminal_logs = []

    for it in items:
        if STOP:
            break

        # 🟩 只取與該公司有關的句子
        text_raw = f"{it.get('title','')}。{it.get('content','')}"
        text_for_score = extract_company_related_text(text_raw, target)
        if not text_for_score.strip():
            continue  # 無公司相關內容 → 跳過

        res = score_text(text_for_score, pos_c, neg_c)
        if abs(res.score) >= SCORE_THRESHOLD:
            filtered.append((it, res))
            trend = "✅ 明日可能大漲" if res.score > 0 else "❌ 明日可能下跌"
            hits_text_lines = [f"  {'+' if w>0 else '-'} {patt}（{note}）" for patt, w, note in res.hits]
            hits_text = "\n".join(hits_text_lines)
            local_entry = f"[{it['id']}] {it.get('title','')}\n{trend}\n命中：\n{hits_text}\n\n內文：\n{text_for_score}\n"
            local_log.append(local_entry)
            truncated_title = first_n_sentences(it.get("title",""), 3)
            terminal_entry_lines = [
                f"[{it['id']}]",
                f"標題：{truncated_title}",
                f"{trend}",
                "命中：",
            ] + hits_text_lines
            terminal_logs.append("\n".join(terminal_entry_lines) + "\n")

    print(f"[info] 過濾後新聞：{len(filtered)} / {len(items)}")
    if not filtered:
        print("[info] 無符合條件的新聞")
        return

    # 顯示前五則命中新聞
    print("\n===== 命中新聞列表（前5則） =====")
    for t in terminal_logs[:5]:
        print(t)

    news_for_llm = [(x[0].get("content") or x[0].get("title") or "") for x in filtered]
    summary = ollama_analyze(news_for_llm, target)
    print("===== Groq 分析 =====")
    print(summary)

    os.makedirs("result", exist_ok=True)
    date_str = datetime.now(TAIWAN_TZ).strftime("%Y%m%d_%H%M%S")
    local_path = f"result/{target}_{date_str}.txt"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write("\n".join(local_log))
        f.write("\n" + "="*60 + "\n")
        f.write(summary + "\n")

    doc_id = datetime.now(TAIWAN_TZ).strftime("%Y%m%d")
    try:
        db.collection(result_col).document(doc_id).set({
            "timestamp": datetime.now(TAIWAN_TZ),
            "result": summary,
        })
    except Exception as e:
        print(f"[error] 寫入 Firebase 失敗：{e}")

# ---------- 主程式 ----------
def main():
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")

    print("\n" + "="*80 + "\n")  # 🟥 分隔線（台積電／鴻海）

    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")

if __name__ == "__main__":
    main()
