# -*- coding: utf-8 -*-
"""
股票新聞分析工具（GitHub Actions + RAG + Groq + Firebase 完整版）
TXT = 每則新聞詳細分析
Firestore = Groq 三行短版輸出
"""

import os
import signal
import regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# ---------- 設定 ----------
SILENT_MODE = True
TAIWAN_TZ = timezone(timedelta(hours=8))

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- 初始化 ----------
if os.path.exists(".env"):
    load_dotenv(".env", override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Firestore ----------
def get_db():
    return firestore.Client()

# ---------- 資料結構 ----------
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

DOCID_RE = re.compile(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$")

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def first_n_sentences(text: str, n: int = 3) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[。\.！!\?？；;])\s*', text.strip())
    return "".join(parts[:n]) + ("..." if len(parts) > n else "")

def parse_docid_time(doc_id: str):
    m = DOCID_RE.match(doc_id or "")
    if not m:
        return None
    ymd = m.group("ymd")
    hms = m.group("hms") or "000000"
    try:
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

# ---------- Token ----------
def load_tokens(db):
    pos, neg = [], []
    for d in db.collection(TOKENS_COLLECTION).stream():
        data = d.to_dict() or {}
        pol = data.get("polarity", "").lower()
        ttype = data.get("type", "substr").lower()
        patt = data.get("pattern", "")
        note = data.get("note", "")
        w = float(data.get("weight", 1.0))

        if pol == "positive":
            pos.append(Token(pol, ttype, patt, w, note))
        elif pol == "negative":
            neg.append(Token(pol, ttype, patt, -abs(w), note))
    return pos, neg

def compile_tokens(tokens: List[Token]):
    compiled = []
    for t in tokens:
        if t.ttype == "regex":
            try:
                compiled.append(("regex", re.compile(t.pattern, re.I), t.weight, t.note, t.pattern))
            except:
                pass
        else:
            compiled.append(("substr", None, t.weight, t.note, t.pattern.lower()))
    return compiled

# ---------- 規則打分 ----------
def score_text(text: str, pos_c, neg_c, target: str = None) -> MatchResult:
    norm = normalize(text)
    score, hits, seen = 0.0, [], set()

    aliases = {
        "台積電": ["台積電", "tsmc", "2330"],
        "鴻海": ["鴻海", "foxconn", "2317", "富士康"],
        "聯電": ["聯電", "umc", "2303"]
    }
    company_pattern = "|".join(re.escape(a) for a in aliases.get(target, []))
    if not re.search(company_pattern, norm):
        return MatchResult(0.0, [])

    for ttype, cre, w, note, patt in pos_c + neg_c:
        key = (patt, note)
        if key in seen:
            continue
        matched = cre.search(norm) if ttype == "regex" else patt in norm
        if matched:
            score += w
            hits.append((patt, w, note))
            seen.add(key)

    return MatchResult(score, hits)

# ---------- RAG：Embedding 篩選 ----------
def rag_select(query: str, texts: List[str], top_k=5):
    if not texts:
        return []
    q_emb = embedder.encode(query, convert_to_tensor=True)
    d_emb = embedder.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, d_emb)[0]
    top = scores.topk(k=min(top_k, len(texts)))
    return [texts[i] for i in top.indices]

# ---------- Groq ----------
def groq_analyze_batch(news: List[str], target: str) -> str:
    if not news:
        return f"""明天{target}股價走勢：不明確 ⚖️
原因：近三日無相關新聞
情緒分數：0"""

    combined = "\n".join(news)

    prompt = f"""
你是專業台股分析師，請嚴格輸出三行：

明天{target}股價走勢：{{上漲／下跌／不明確}} {{符號}}
原因：一句原因
情緒分數：-10~10

新聞內容：
{combined}
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"""明天{target}股價走勢：不明確 ⚖️
原因：Groq 失敗 {e}
情緒分數：0"""

# ---------- TXT ----------
def dump_detailed_news(target, today, rows):
    fname = os.path.join(RESULTS_DIR, f"result_{today.strftime('%Y%m%d')}.txt")
    with open(fname, "a", encoding="utf-8-sig") as f:
        f.write(f"\n📰 {target} 新聞詳細分析：\n\n")
        for docid, key, title, res, w in rows:
            f.write(f"[{docid}#{key}] ({w:.2f}x, 分數={res.score:+.2f}) {first_n_sentences(title)}\n")
            for patt, ww, note in res.hits:
                sign = "+" if ww > 0 else "-"
                f.write(f"   {sign} {patt}（{note}）\n")
            f.write("\n")

# ---------- 主流程 ----------
def analyze_target(db, collection: str, target: str, result_field: str):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()

    all_news = []
    raw_texts = []

    aliases = {
        "台積電": ["台積電", "tsmc", "2330"],
        "鴻海": ["鴻海", "foxconn", "2317", "富士康"],
        "聯電": ["聯電", "umc", "2303"]
    }
    company_keywords = aliases[target]

    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        if (today - dt.date()).days > 2:
            continue

        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue

            title = v.get("title", "")
            content = v.get("content", "")
            full = f"{title} {content}"

            if not any(key.lower() in full.lower() for key in company_keywords):
                continue

            res = score_text(full, pos_c, neg_c, target)
            if not res.hits:
                continue

            all_news.append((d.id, k, title, res, 1.0))
            raw_texts.append(full)

    # TXT
    if all_news:
        dump_detailed_news(target, today, all_news)

    # RAG 選新聞
    query = f"{target} 明日股價走勢"
    rag_news = rag_select(query, raw_texts, 5)

    # Groq
    summary = groq_analyze_batch(rag_news, target)
    print(summary + "\n")

    # Firebase
    try:
        db.collection(result_field).document(today.strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "result": summary
        })
    except Exception as e:
        print("[warning] Firestore 寫入失敗:", e)

# ---------- main ----------
def main():
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
