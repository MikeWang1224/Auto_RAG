# -*- coding: utf-8 -*-
"""
股票新聞分析工具（GitHub Actions 優化版 + 詳細輸出）
✅ 批次 Groq 呼叫
✅ Firestore 拉取與 scoring 加計時
✅ 限制 top_n 篇新聞
✅ Log 計時，方便 GitHub Runner 排查
✅ 新增：完整詳細新聞評分輸出（加權、分數、衝擊、token 命中 note）
"""

import os, signal, regex as re, time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 設定 ----------
SILENT_MODE = True
TAIWAN_TZ = timezone(timedelta(hours=8))
TOP_N = 10  # 每家公司只分析前 N 篇新聞，避免過久

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

# ---------- Groq 批次分析 ----------
def groq_analyze_batch(news_with_scores: List[Tuple[str, float]], target: str, price_change: str = "") -> str:
    if not news_with_scores:
        reason_text = f"近三日無相關新聞。今日漲跌：{price_change}" if price_change else "近三日無相關新聞"
        return f"明天{target}股價走勢：不明確 ⚖️\n原因：{reason_text}\n情緒分數：0"

    combined = "\n".join(f"{i+1}. ({s:+.2f}) {t}" for i, (t, s) in enumerate(news_with_scores))

    avg_score = sum(s for _, s in news_with_scores) / len(news_with_scores)

    prompt_text = f"""
你是一位金融新聞分析員。
請閱讀以下關於「{target}」最近三天的新聞摘要，
整體平均情緒分數為 {avg_score:+.2f}：

{combined}

請輸出格式如下：
明天{target}股價走勢：{{上漲／下跌／不明確}}（附符號）
原因：{{一句總結理由}}
情緒分數：{{整數（-10~10）}}
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是專業台股分析師。"},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.2,
            max_tokens=200,
            timeout=25,
        )
        ans = resp.choices[0].message.content.strip()
        ans = re.sub(r"\s+", " ", ans)

        m_trend = re.search(r"(上漲|下跌|不明確|微漲|微跌)", ans)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲":"🔼","微漲":"↗️","微跌":"↘️","下跌":"🔽","不明確":"⚖️"}

        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+?)(?:情緒分數|$)", ans)
        reason = m_reason.group(1).strip() if m_reason else f"市場觀望。今日漲跌：{price_change}"

        m_score = re.search(r"情緒分數[:：]?\s*(-?\d+)", ans)
        mood_score = int(m_score.group(1)) if m_score else 0

        return f"明天{target}股價走勢：{trend} {symbol_map.get(trend,'')}\n原因：{reason}\n情緒分數：{mood_score:+d}"

    except Exception as e:
        return f"明天{target}股價走勢：持平 ⚖️\n原因：Groq分析失敗({e})\n情緒分數：0"

# ---------- ★ 新增：詳細輸出 ----------
def dump_detailed_news(target: str, today, top_news: List[Tuple]):
    fname = f"result_{today.strftime('%Y%m%d')}.txt"
    with open(fname, "a", encoding="utf-8") as f:
        f.write(f"📰 {target} 近期重點新聞（含衝擊）:\n\n")
        for docid, key, title, res, weight in top_news:
            raw_score = res.score
            impact = 1.00  # 你目前邏輯固定 impact = 1.0
            f.write(
                f"[{docid}#{key}] ({weight:.2f}x, 分數={raw_score:+.2f}, 衝擊={impact:.2f}) "
                f"{first_n_sentences(title)}\n"
            )
            for patt, w, note in res.hits:
                sign = "+" if w > 0 else "-"
                f.write(f"   {sign} {patt}（{note}）\n")
            f.write("\n")

# ---------- 主分析 ----------
def analyze_target(db, collection: str, target: str, result_field: str):
    t0 = time.time()
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    t1 = time.time()
    print(f"[計時] Token 載入耗時: {t1-t0:.2f}s")

    today = datetime.now(TAIWAN_TZ).date()
    filtered, weighted_scores = [], []
    price_change = ""

    # Firestore 拉取
    t_start = time.time()
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        news_date = dt.date()
        delta_days = (today - news_date).days
        if delta_days > 2:
            continue

        day_weight = {0:1.0, 1:0.85, 2:0.7}.get(delta_days,0.7)
        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            if not price_change:
                price_change = v.get("price_change", "")
            title, content = v.get("title", ""), v.get("content", "")
            full = title + " " + content
            res = score_text(full, pos_c, neg_c, target)
            if not res.hits:
                continue
            token_weight = 1.0 + min(len(res.hits) * 0.05, 0.3)
            total_weight = day_weight * token_weight
            filtered.append((d.id, k, title, res, total_weight))
            weighted_scores.append(res.score * total_weight)
    t_end = time.time()
    print(f"[計時] Firestore 拉取與 scoring 耗時: {t_end-t_start:.2f}s")

    if not filtered:
        print(f"{target}：近三日無新聞，交由 Groq 判斷。\n")
        summary = groq_analyze_batch([], target, price_change)
    else:
        filtered.sort(key=lambda x: abs(x[3].score * x[4]), reverse=True)
        top_news = filtered[:TOP_N]

        news_with_scores = [(t, res.score * weight) for _, _, t, res, weight in top_news]
        summary = groq_analyze_batch(news_with_scores, target, price_change)

        # ★ 完整詳細輸出
        dump_detailed_news(target, today, top_news)

        # 總結加在最後
        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(summary + "\n\n")

    print(summary + "\n")

    # Firestore 寫回
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
        print("🚀 開始分析台股焦點股（GitHub Actions 版）...\n")
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
