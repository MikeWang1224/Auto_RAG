# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
完全可跑版（短期預測特化） - Context-aware + 背離偵測
⬆️ 修正：今天新聞 100% 會被抓到
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

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

SENSITIVE_WORDS = {
    "法說": 1.5, "財報": 1.4, "新品": 1.3, "合作": 1.3,
    "併購": 1.4, "投資": 1.3, "停工": 1.6, "下修": 1.5,
    "利空": 1.5, "爆料": 1.4, "營收": 1.3, "展望": 1.2,
}

STOP = False
def _sigint_handler(signum, frame):
    global STOP
    STOP = True
    print("\n[info] 偵測到 Ctrl+C，將安全停止…")
signal.signal(signal.SIGINT, _sigint_handler)

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
    """
    修正：如果解析 docID 時間失敗 → 自動視為今天
    避免今天新聞被忽略
    """
    m = re.match(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$", doc_id or "")
    if not m:
        return datetime.now(TAIWAN_TZ)

    ymd, hms = m.group("ymd"), m.group("hms") or "000000"
    try:
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return datetime.now(TAIWAN_TZ)

def parse_price_change(val):
    if not isinstance(val, str) or not val.strip():
        return 0.0
    m = re.search(r"\((-?\d*\.?\d+)%\)", val)
    if m:
        return float(m.group(1))
    m = re.search(r"([-+]?\d*\.?\d+)%", val)
    if m:
        return float(m.group(1))
    return 0.0

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
        if pol == "positive": pos.append(Token(pol, ttype, patt, w, note))
        elif pol == "negative": neg.append(Token(pol, ttype, patt, -abs(w), note))
    return pos, neg

def compile_tokens(tokens: List[Token]):
    compiled = []
    for t in tokens:
        if t.ttype == "regex":
            try: compiled.append(("regex", re.compile(t.pattern, re.I), t.weight, t.note, t.pattern))
            except: continue
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
        "聯電": ["聯電", "umc", "2303"],
    }
    company_pattern = "|".join(re.escape(a) for a in aliases.get(target, []))
    if not re.search(company_pattern, norm):
        return MatchResult(0.0, [])

    for ttype, cre, w, note, patt in pos_c + neg_c:
        key = (patt, note)
        if key in seen: continue
        matched = cre.search(norm) if ttype == "regex" else patt in norm
        if matched:
            score += w
            hits.append((patt, w, note))
            seen.add(key)

    return MatchResult(score, hits)

def adjust_score_for_context(text: str, base_score: float) -> float:
    if not text or base_score == 0: return base_score
    norm = text.lower()

    neutral_phrases = ["重申", "符合預期", "預期內", "中性看待", "無重大影響", "持平", "未變"]
    if any(p in norm for p in neutral_phrases):
        base_score *= 0.4

    positive_boost = ["創新高", "倍增", "大幅成長", "獲利暴增", "報喜"]
    negative_boost = ["暴跌", "下滑", "虧損", "停工", "下修", "裁員", "警訊"]

    if any(p in norm for p in positive_boost): base_score *= 1.3
    if any(p in norm for p in negative_boost): base_score *= 1.3

    return base_score

# ---------- 背離偵測 ----------
def detect_divergence(avg_score: float, top_news):
    key_news = top_news[:5]
    price_moves = []
    strength = []

    for _, _, _, res, weight, price_change in key_news:
        pc = price_change if price_change is not None else 0.0
        price_moves.append(pc * weight)
        strength.append(abs(res.score * weight))

    if not price_moves:
        return "無足夠資料判斷背離。"

    avg_strength = sum(strength) / len(strength)
    if avg_strength < 0.4:
        return "新聞力道偏弱，無明顯背離。"

    avg_price_move = sum(price_moves) / len(price_moves)

    STRONG = 0.7
    MEDIUM = 0.35

    if avg_score > STRONG and avg_price_move < -0.2:
        return "新聞偏強多，但股價顯著下跌，屬正向背離（可能短線反彈）。"

    if avg_score > MEDIUM and avg_price_move < -0.5:
        return "新聞多方略強，股價卻走弱，可能正向背離。"

    if avg_score < -STRONG and avg_price_move > 0.2:
        return "新聞偏強空，但股價顯著上漲，屬負向背離（可能短線回檔）。"

    if avg_score < -MEDIUM and avg_price_move > 0.5:
        return "新聞空方略強，股價卻上漲，可能負向背離。"

    return "股價走勢與新聞情緒一致，無明顯背離。"

# ---------- Groq ----------
def groq_analyze(news_list, target, avg_score, divergence_note=None):
    if not news_list:
        return f"隔日{target}股價走勢：不明確 ⚖️\n原因：近三日無相關新聞"

    combined = "\n".join(f"{i+1}. ({s:+.2f}) {t}" for i, (t, s) in enumerate(news_list))
    divergence_text = f"\n此外，背離判斷：{divergence_note}" if divergence_note else ""

    prompt = f"""
你是一位專業的台股金融分析師，請根據以下「{target}」近三日新聞摘要，
依情緒分數與內容趨勢，嚴格推論隔日股價方向。
請只輸出「走勢 + 原因」，不要輸出情緒分數。

請用以下格式：
隔日{target}股價走勢：{{上漲／微漲／微跌／下跌／不明確}}（附符號）
原因：{{一句 55 字內，只描述新聞與情緒方向}}
{divergence_text}

整體平均情緒分數：{avg_score:+.2f}
新聞摘要（含分數）：
{combined}
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是台股量化分析員，需依情緒分數產生明確結論，但輸出不能包含情緒分數。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=220,
        )

        ans = resp.choices[0].message.content.strip()

        ans = re.sub(r"情緒分數[:：]\s*-?\d+(\.\d+)?", "", ans)
        ans = re.sub(r"\n{2,}", "\n", ans).strip()

        m_trend = re.search(r"(上漲|微漲|微跌|下跌|不明確)", ans)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲": "🔼", "微漲": "↗️", "微跌": "↘️", "下跌": "🔽", "不明確": "⚖️"}

        m_reason = re.search(r"(?:原因|理由)[:：]\s*(.*)", ans)
        reason = m_reason.group(1).strip() if m_reason else ""

        return f"下個預測{target}股價走勢：{trend} {symbol_map.get(trend, '')}\n原因：{reason}"

    except Exception as e:
        return f"隔日{target}股價走勢：不明確 ⚖️\n原因：Groq分析失敗({e})"

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()

    filtered = []
    seen_news = set()

    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        delta_days = (today - dt.date()).days

        # ⭐ 今天新聞必抓！
        if delta_days < 0:
            delta_days = 0

        if delta_days > 2:
            continue

        day_weight = 1.0 if delta_days == 0 else 0.85 if delta_days == 1 else 0.7
        data = d.to_dict() or {}

        for k, v in data.items():
            if not isinstance(v, dict):
                continue

            title, content = v.get("title", ""), v.get("content", "")

            full_raw = f"{title}|{content}"
            if full_raw in seen_news:
                continue
            seen_news.add(full_raw)

            price_raw = v.get("price_change", "")
            price_change = parse_price_change(price_raw)

            full = f"{title} {content} 股價變動：{price_raw}"

            res = score_text(full, pos_c, neg_c, target)
            if not res.hits:
                continue

            adj_score = adjust_score_for_context(full, res.score)
            token_weight = 1.0 + min(len(res.hits) * 0.05, 0.3)
            impact = 1.0 + sum(w * 0.05 for k_sens, w in SENSITIVE_WORDS.items() if k_sens in full)
            total_weight = day_weight * token_weight * impact

            filtered.append((d.id, k, title, res, total_weight, price_change))

    if not filtered:
        print(f"{target}：近三日無新聞，交由 Groq 判斷。\n")
        summary = groq_analyze([], target, 0)
    else:
        filtered.sort(key=lambda x: abs(x[3].score * x[4]), reverse=True)
        top_news = filtered[:10]

        print(f"\n📰 {target} 近期重點新聞（含衝擊）：")
        for docid, key, title, res, weight, price_change in top_news:
            impact = sum(w for k_sens, w in SENSITIVE_WORDS.items() if k_sens in title)
            print(f"[{docid}#{key}] ({weight:.2f}x, 分數={res.score:+.2f}, 衝擊={1+impact/10:.2f}) {title} | 股價變動：{price_change}")
            for p, w, n in res.hits:
                print(f"   {'+' if w > 0 else '-'} {p}（{n}）")

        news_with_scores = [(f"{t} 股價變動：{pc}", res.score * weight) for _, _, t, res, weight, pc in top_news]
        avg_score = sum(s for _, s in news_with_scores) / len(news_with_scores)
        divergence_note = detect_divergence(avg_score, top_news)
        summary = groq_analyze(news_with_scores, target, avg_score, divergence_note)

        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            for docid, key, title, res, weight, price_change in top_news:
                hits_text = "\n".join([f"  {'+' if w > 0 else '-'} {p}（{n}）" for p, w, n in res.hits])
                f.write(f"[{docid}#{key}]（{weight:.2f}x）\n標題：{first_n_sentences(title)}\n股價變動：{price_change}\n命中：\n{hits_text}\n\n")
            f.write(f"★ 背離判斷：{divergence_note}\n")
            f.write(f"下個預測股價走勢：{summary}\n\n")

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
        print("🚀 開始分析台股焦點股（完全可跑版）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
