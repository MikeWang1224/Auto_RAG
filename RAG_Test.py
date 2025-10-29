# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
最終版：
✅ 僅分析「今日」新聞
✅ 只列入分數 > 1.5 的新聞
✅ 命中 token 不重複
✅ Firestore 寫回結果
✅ Groq 失敗自動持平
✅ 執行速度最佳化
"""

import os, signal, regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 設定 ----------
SILENT_MODE = True  # ✅ 設為 True 時不顯示 🚀 開始分析... 的提示
SCORE_THRESHOLD = 1.5
TAIWAN_TZ = timezone(timedelta(hours=8))
MAX_DISPLAY_NEWS = 5

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

STOP = False
def _sigint_handler(signum, frame):
    global STOP
    STOP = True
    print("\n[info] 偵測到 Ctrl+C，將安全停止…")
signal.signal(signal.SIGINT, _sigint_handler)

# ---------- 初始化 ----------
if os.path.exists(".env"):
    load_dotenv(".env", override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
DOCID_RE = re.compile(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$")

def get_db():
    return firestore.Client()

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
def load_tokens(db) -> Tuple[List[Token], List[Token]]:
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
                continue
        else:
            compiled.append(("substr", None, t.weight, t.note, t.pattern.lower()))
    return compiled

# ---------- Scoring ----------
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

# ---------- Groq ----------
def groq_analyze(news_list: List[str], target: str) -> str:
    if not news_list:
        return f"明天{target}股價走勢：持平 ⚖️\n原因：今日無顯著新聞"
    combined = "\n".join(f"{i+1}. {t}" for i, t in enumerate(news_list[:10]))
    prompt = f"""你是一位台股分析師。根據以下{target}相關新聞，請簡短判斷明天{target}股價走勢：
請僅回傳：
明天{target}股價走勢：<上漲 / 下跌 / 不明確>
原因：<一句話40字內>

{combined}
"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是專業股市分析師，回答簡潔準確。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=20,
        )
        ans = resp.choices[0].message.content.strip()
        ans = re.sub(r"\s+", " ", ans)
        m_trend = re.search(r"(上漲|下跌|不明確)", ans)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲": "🔼", "下跌": "🔽", "不明確": "⚖️"}
        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+)", ans)
        reason = m_reason.group(1) if m_reason else "市場觀望"
        return f"明天{target}股價走勢：{trend} {symbol_map.get(trend,'')}\n原因：{reason[:40]}"
    except Exception as e:
        return f"明天{target}股價走勢：持平 ⚖️\n原因：Groq分析失敗({e})"

# ---------- 主分析 ----------
def analyze_target(db, collection: str, target: str, result_field: str):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)

    today_str = datetime.now(TAIWAN_TZ).strftime("%Y%m%d")
    items, filtered = [], []

    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt or dt.strftime("%Y%m%d") != today_str:
            continue
        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            title, content = v.get("title", ""), v.get("content", "")
            full = title + " " + content
            res = score_text(full, pos_c, neg_c, target)
            if abs(res.score) <= SCORE_THRESHOLD or not res.hits:
                continue
            filtered.append((d.id, k, title, res))
            trend = "✅ 明日可能大漲" if res.score > 0 else "❌ 明日可能下跌"
            hits_text = "\n".join([f"  {'+' if w>0 else '-'} {p}（{n}）" for p, w, n in res.hits])
            print(f"[{d.id}#{k}]\n標題：{first_n_sentences(title)}\n{trend}\n命中：\n{hits_text}\n")

    if not filtered:
        print(f"{target}：持平 ⚖️（今日無顯著新聞）\n")
        return

    news_texts = [t for _, _, t, _ in filtered]
    summary = groq_analyze(news_texts, target)
    print(summary + "\n")

    try:
        db.collection(result_field).document(today_str).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "result": summary,
        })
    except Exception as e:
        print(f"[warning] Firestore 寫回失敗：{e}")

# ---------- 主程式 ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股（僅今日新聞，分數 > 1.5）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    if not SILENT_MODE:
        print("="*70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    if not SILENT_MODE:
        print("="*70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
