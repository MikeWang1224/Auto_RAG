# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海）
靜默模式：不顯示多餘訊息，只輸出前五則命中新聞與 Groq 結果。
"""


import os, signal, regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 設定 ----------
SILENT_MODE = True        # ✅ 完全靜默模式
MAX_DISPLAY_NEWS = 5      # ✅ 終端最多顯示前 5 則新聞

def log(msg: str):
    if not SILENT_MODE:
        print(msg)

# ---------- 讀 .env ----------
if os.path.exists(".env"):
    load_dotenv(".env", override=True)
else:
    load_dotenv(override=True)

# ---------- 常數 ----------
TOKENS_COLLECTION = os.getenv("FIREBASE_TOKENS_COLLECTION", "bull_tokens")
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "3.0"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "2"))
TAIWAN_TZ = timezone(timedelta(hours=8))

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
    joined = "".join(parts[:n])
    if not re.search(r'[。\.！!\?？；;]\s*$', joined):
        joined += "..."
    return joined

def parse_docid_time(doc_id: str):
    m = DOCID_RE.match(doc_id)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ymd")+m.group("hms"), "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

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
        if not patt or pol not in ("positive", "negative"):
            continue
        (pos if pol == "positive" else neg).append(Token(pol, ttype, patt, w, note))
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
            title, content = str(v.get("title") or ""), str(v.get("content") or "")
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
    return "\n".join(f"新聞 {i}：\n{shorten_text(t)}\n" for i, t in enumerate(news_items, 1))

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
        m_trend = re.search(r"(上漲|下跌|不明確)", cleaned)
        trend = m_trend.group(1) if m_trend else "不明確"
        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+)", cleaned)
        reason_text = m_reason.group(1) if m_reason else cleaned
        sentences = re.split(r"[。.!！；;]", reason_text)
        short_reason = "，".join(sentences[:2]).strip()
        short_reason = re.sub(r"\s+", " ", short_reason)[:40].strip("，,。")
        return f"明天{target}股價走勢：{trend}\n原因：{short_reason}"
    except Exception as e:
        return f"[error] Groq 呼叫失敗：{e}"

# ---------- 分析通用函數 ----------
def analyze_target(db, news_col: str, target: str, result_col: str):
    pos, neg = load_tokens(db, TOKENS_COLLECTION)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    items = load_news_items(db, news_col, LOOKBACK_DAYS)
    if not items:
        return

    filtered, local_log, terminal_logs = [], [], []
    for it in items:
        if STOP:
            break
        res = score_text(it.get("content") or it.get("title") or "", pos_c, neg_c)
        if abs(res.score) >= SCORE_THRESHOLD:
            filtered.append((it, res))
            trend = "✅ 明日可能大漲" if res.score > 0 else "❌ 明日可能下跌"
            hits_text_lines = [f"  {'+' if w>0 else '-'} {patt}（{note}）" for patt, w, note in res.hits]
            local_log.append(f"[{it['id']}] {it.get('title','')}\n{trend}\n命中：\n" + "\n".join(hits_text_lines))
            truncated_title = first_n_sentences(it.get("title",""), 3)
            terminal_logs.append(f"[{it['id']}]\n標題：{truncated_title}\n{trend}\n命中：\n" + "\n".join(hits_text_lines) + "\n")

    # --- 只顯示前五則 ---
    for t in terminal_logs[:MAX_DISPLAY_NEWS]:
        print(t)

    summary = ollama_analyze([(x[0].get("content") or x[0].get("title") or "") for x in filtered], target)
    print(summary)

    os.makedirs("result", exist_ok=True)
    local_path = f"result/{target}_{datetime.now(TAIWAN_TZ).strftime('%Y%m%d_%H%M%S')}.txt"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write("\n".join(local_log))
        f.write("\n" + "="*60 + "\n")
        f.write(summary + "\n")

    try:
        db.collection(result_col).document(datetime.now(TAIWAN_TZ).strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ),
            "result": summary,
        })
    except Exception as e:
        log(f"[error] 寫入 Firebase 失敗：{e}")

# ---------- 主程式 ----------
def main():
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("\n" + "="*70 + "\n")   # ✅ 分隔線（台積電 ↔ 鴻海）
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")

if __name__ == "__main__":
    main()
