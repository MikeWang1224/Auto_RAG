# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
加強版（含延遲效應時間窗 + 高權重新聞前五則優先）：
✅ 分析「今日 + 昨日」新聞（2天延遲效應）
✅ 今日新聞權重 = 1.0、昨日 = 0.7
✅ 依加權後分數絕對值排序取前 5 則新聞送 Groq
✅ 終端印出給 Groq 的新聞標題（方便確認）
✅ Groq 永遠會分析（即使無新聞）
✅ 若 Groq 回「不明確」，依加權平均分數自動微調
✅ 股票間輸出用 ======= 分隔
"""

import os, signal, regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 設定 ----------
SILENT_MODE = True
TAIWAN_TZ = timezone(timedelta(hours=8))
SCORE_THRESHOLD = 1.5

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
    combined = "\n".join(f"{i+1}. {t}" for i, t in enumerate(news_list[:10]))
    prompt = f"""你是一位專業台股分析師。根據以下{target}的近期新聞內容，
請判斷明天{target}股價最可能的方向（上漲或下跌，如真的難判斷再選不明確）：
回傳格式如下：
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
            temperature=0.1,
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

    today = datetime.now(TAIWAN_TZ).date()
    filtered = []
    weighted_scores = []

    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        news_date = dt.date()
        delta_days = (today - news_date).days
        if delta_days > 1:
            continue
        weight = 1.0 if delta_days == 0 else 0.7

        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            title, content = v.get("title", ""), v.get("content", "")
            full = title + " " + content
            res = score_text(full, pos_c, neg_c, target)
            if not res.hits:
                continue
            filtered.append((d.id, k, title, res, weight))
            weighted_scores.append(res.score * weight)

    # ✅ 若完全沒有新聞
    if not filtered:
        print(f"{target}：近兩日無新聞，交由 Groq 判斷。\n")
        summary = groq_analyze(["近兩日無相關新聞，請依市場情緒估計。"], target)
    else:
        # ✅ 取前五則（同 Groq）
        filtered.sort(key=lambda x: abs(x[3].score * x[4]), reverse=True)
        top_news = filtered[:5]
        news_texts = [t for _, _, t, _, _ in top_news]

        # ✅ 給 Groq 的新聞列表（新增這段）
        print(f"\n📰 給 Groq 分析的 {target} 前五則新聞：")
        for i, (docid, key, title, res, weight) in enumerate(top_news, 1):
            print(f"[{docid}#{key}] ({weight:.1f}x, 分數={res.score:.2f}) {title}")

        # --- 輸出 .txt ---
        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            for docid, key, title, res, weight in top_news:
                trend = "✅ 明日可能大漲" if res.score > 0 else "❌ 明日可能下跌"
                hits_text = "\n".join([f"  {'+' if w>0 else '-'} {p}（{n}）" for p, w, n in res.hits])
                f.write(f"[{docid}#{key}]（{weight:.1f}x）\n標題：{first_n_sentences(title)}\n{trend}\n命中：\n{hits_text}\n\n")

        summary = groq_analyze(news_texts, target)

        # ✅ 根據平均分數微調
        if weighted_scores:
            avg_score = sum(weighted_scores) / len(weighted_scores)
            if avg_score > 1.5:
                summary = re.sub(r"不明確", "上漲", summary)
            elif avg_score < -1.5:
                summary = re.sub(r"不明確", "下跌", summary)

    print(summary + "\n")

    try:
        db.collection(result_field).document(today.strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "result": summary,
        })
    except Exception as e:
        print(f"[warning] Firestore 寫回失敗：{e}")

# ---------- 主程式 ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股（2天延遲效應版 + 前五則高權重新聞）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
