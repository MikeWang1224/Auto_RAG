# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
Firebase 簡短摘要版（TXT 保留完整分析）
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
    "法說": 1.5,
    "財報": 1.4,
    "新品": 1.3,
    "合作": 1.3,
    "併購": 1.4,
    "投資": 1.3,
    "停工": 1.6,
    "下修": 1.5,
    "利空": 1.5,
    "爆料": 1.4,
    "營收": 1.3,
    "展望": 1.2,
}

HARD_WEIGHTS_POS = {
    "財報": 1.5,
    "法說": 1.5,
    "展望": 1.5,
    "資本支出": 1.5,
    "訂單": 1.2,
    "擴產": 1.2,
    "爆單": 1.2,
    "漲價": 1.2,
}
HARD_WEIGHTS_NEG = {
    "停工": -1.5,
    "裁員": -1.5,
    "虧損": -1.5,
    "下修": -1.5,
}

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
    m = re.match(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$", doc_id or "")
    if not m:
        return None
    ymd, hms = m.group("ymd"), m.group("hms") or "000000"
    try:
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

def parse_price_change(raw: str) -> float:
    if not raw:
        return 0.0
    m = re.search(r"\(([-+]?[\d\.]+)%\)", raw)
    if not m:
        return 0.0
    try:
        return float(m.group(1)) / 100.0
    except:
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

def score_text(text: str, pos_c, neg_c, target: str = None) -> MatchResult:
    norm = normalize(text)
    score, hits, seen = 0.0, [], set()
    aliases = {"台積電": ["台積電", "tsmc", "2330"],
               "鴻海": ["鴻海", "foxconn", "2317", "富士康"],
               "聯電": ["聯電", "umc", "2303"]}
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

def adjust_score_for_context(text: str, base_score: float) -> float:
    if not text or base_score == 0:
        return base_score

    norm = text.lower()
    neutral_phrases = ["重申", "符合預期", "預期內", "中性看待", "無重大影響", "持平", "未變"]
    if any(p in norm for p in neutral_phrases):
        base_score *= 0.4

    positive_boost = ["創新高", "倍增", "大幅成長", "獲利暴增", "報喜"]
    negative_boost = ["暴跌", "下滑", "虧損", "停工", "下修", "裁員", "警訊"]
    if any(p in norm for p in positive_boost):
        base_score *= 1.3
    if any(p in norm for p in negative_boost):
        base_score *= 1.3

    return base_score

def adjust_by_market(avg_score: float, today_change: float) -> float:
    if today_change >= 0.03:
        return avg_score + 0.5
    if today_change <= -0.03:
        return avg_score - 0.5
    if today_change >= 0.01:
        return avg_score + 0.2
    if today_change <= -0.01:
        return avg_score - 0.2
    return avg_score

def detect_divergence(avg_score: float, today_change: float) -> str:
    if avg_score > 1.0 and today_change < -0.02:
        return "利多不漲（疑似主力出貨）"
    if avg_score < -1.0 and today_change > 0.02:
        return "利空不跌（可能有隱性利多）"
    return "無明顯背離"

def decide_by_hard_rules(news_list: List[Tuple[str, float]], today_change: float, full_texts: List[str] = None, adjusted_avg: float = None, divergence: str = None) -> str:
    n = len(news_list)
    if n == 0:
        return "明天股價走勢：不明確 ⚖️\n原因：近三日無相關新聞\n情緒分數：0"

    contributions = []
    reason_lines = []
    for idx, (title, weighted_score) in enumerate(news_list):
        base = 1.0 if weighted_score > 0 else (-1.0 if weighted_score < 0 else 0.0)
        add = 0.0
        txt = (full_texts[idx] if full_texts and idx < len(full_texts) else title).lower()

        for kw, v in HARD_WEIGHTS_POS.items():
            if kw in txt:
                add += v
                reason_lines.append(f"新聞[{idx+1}] 包含正向關鍵詞「{kw}」，加成 {v:+.2f}")
                break
        for kw, v in HARD_WEIGHTS_NEG.items():
            if kw in txt:
                add += v
                reason_lines.append(f"新聞[{idx+1}] 包含負向關鍵詞「{kw}」，加成 {v:+.2f}")
                break

        contrib = base + add
        contributions.append(contrib)
        reason_lines.append(f"新聞[{idx+1}]：標題/內容摘要「{first_n_sentences(title,1)}」，基礎貢獻 {base:+.2f}，加權後 {contrib:+.2f}")

    total_score = sum(contributions)
    standardized = total_score / (n + 1)

    if adjusted_avg is not None:
        standardized = adjusted_avg
        reason_lines.append(f"（已套用市場漲跌幅調整，使用調整後分數 {standardized:+.2f}）")

    if standardized >= 2.5:
        impact = 1; trend = "上漲"; symbol = "🔼"
    elif standardized >= 1.0:
        impact = 2; trend = "微漲"; symbol = "↗️"
    elif standardized > -1.0:
        impact = 3; trend = "微跌"; symbol = "↘️"
    else:
        impact = 4; trend = "下跌"; symbol = "🔽"

    pct = round(today_change * 100, 2)
    trend_today = "上漲" if today_change > 0 else "下跌" if today_change < 0 else "平盤"

    dir_sign = 1 if standardized > 0 else (-1 if standardized < 0 else 0)
    today_sign = 1 if today_change > 0 else (-1 if today_change < 0 else 0)
    if dir_sign != 0 and today_sign != 0:
        if dir_sign == today_sign:
            market_effect = "今日走勢與新聞方向同向，市場走勢強化新聞信號。"
        else:
            market_effect = "今日走勢與新聞方向相反，市場走勢可能已提前消化或抵銷新聞影響。"
    else:
        market_effect = "今日走勢或新聞方向中性，無明顯強化/抵銷判斷。"

    mood_score = max(-10, min(10, int(round(standardized * 3))))

    if divergence:
        reason_lines.append(f"市場背離檢測：{divergence}")

    detail_reason = "\n".join(reason_lines)
    summary_reason = f"標準化分數 {standardized:+.2f}；{market_effect} (今日漲跌 {trend_today} {pct}%)"

    final_text = (
        f"明天股價走勢：{trend} {symbol}\n"
        f"原因：{summary_reason}\n"
        f"細節：\n{detail_reason}\n"
        f"情緒分數：{mood_score:+d}"
    )
    return final_text, trend, mood_score

# ---------- 新增短版摘要 ----------
def build_short_summary(target_name: str, trend: str, reason: str, sentiment_score: float) -> str:
    arrow = "🔼" if trend in ["上漲", "微漲"] else "🔽" if trend in ["下跌", "微跌"] else "➡️"
    return f"明天{target_name}股價走勢：{trend} {arrow} 原因：{reason} 情緒分數：{round(sentiment_score)}"

# ---------- Groq 判斷（硬規則） ----------
def groq_analyze(news_list, target, avg_score, today_change, adjusted_avg=None, divergence=None):
    full_texts = [t for t, _ in news_list]
    full_result, trend, mood_score = decide_by_hard_rules(news_list, today_change, full_texts, adjusted_avg=adjusted_avg, divergence=divergence)
    # 簡短原因：選 top 2 消極新聞標題拼接
    reason_snippets = [t for t, s in news_list if s < 0][:2]
    short_reason = "、".join([first_n_sentences(r,1) for r in reason_snippets]) or "近期新聞無明顯影響"
    short_summary = build_short_summary(target, trend, short_reason, mood_score)
    return full_result, short_summary

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()

    filtered, weighted_scores = [], []
    today_price_change = 0.0

    # ---------- 先掃一次 collection 取得今日 price_change ----------
    try:
        for d in db.collection(collection).stream():
            dt = parse_docid_time(d.id)
            if not dt or dt.date() != today:
                continue
            data = d.to_dict() or {}
            for k, v in data.items():
                if isinstance(v, dict) and "price_change" in v:
                    today_price_change = parse_price_change(v.get("price_change"))
                    break
            if today_price_change != 0.0:
                break
    except Exception:
        today_price_change = 0.0

    # ---------- 新聞打分流程 ----------
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt or (today - dt.date()).days > 2:
            continue
        day_weight = 1.0 if (today - dt.date()).days == 0 else 0.85 if (today - dt.date()).days == 1 else 0.7
        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            title, content = v.get("title", ""), v.get("content", "")
            full = title + " " + content
            res = score_text(full, pos_c, neg_c, target)
            if not res.hits:
                continue
            adj_score = adjust_score_for_context(full, res.score)
            token_weight = 1.0 + min(len(res.hits) * 0.05, 0.3)
            impact = 1.0 + sum(w * 0.05 for k_sens, w in SENSITIVE_WORDS.items() if k_sens in full)
            total_weight = day_weight * token_weight * impact
            filtered.append((d.id, k, title, full, res, total_weight))
            weighted_scores.append(adj_score * total_weight)

    if not filtered:
        full_text, short_text = groq_analyze([], target, 0, today_price_change)
    else:
        seen_text = set()
        top_news = []
        for docid, key, title, full, res, weight in sorted(filtered, key=lambda x: abs(x[4].score * x[5]), reverse=True):
            news_text = normalize(full)
            if news_text in seen_text:
                continue
            seen_text.add(news_text)
            top_news.append((docid, key, title, res, weight, full))
            if len(top_news) >= 10:
                break

        print(f"\n📰 {target} 近期重點新聞（含衝擊）:")
        for docid, key, title, res, weight, full in top_news:
            impact_val = sum(w for k_sens, w in SENSITIVE_WORDS.items() if k_sens in title)
            print(f"[{docid}#{key}] ({weight:.2f}x, 分數={res.score:+.2f}, 衝擊={1+impact_val/10:.2f}) {title}")
            for p, w, n in res.hits:
                sign = "+" if w>0 else "-"
                print(f"   {sign} {p}（{n}）")

        news_with_scores = [(title, res.score * weight) for _, _, title, res, weight, _ in top_news]

        avg_score = sum(s for _, s in news_with_scores) / len(news_with_scores)
        adjusted_avg = adjust_by_market(avg_score, today_price_change)
        divergence = detect_divergence(avg_score, today_price_change)

        full_text, short_text = groq_analyze(news_with_scores, target, avg_score, today_price_change, adjusted_avg=adjusted_avg, divergence=divergence)

        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            f.write(f"今日漲跌 (程式讀取)：{round(today_price_change*100,2)}%\n")
            f.write(f"avg_score (原始)：{avg_score:+.4f}\n")
            f.write(f"avg_score (調整後)：{adjusted_avg:+.4f}\n")
            f.write(f"背離檢測：{divergence}\n")
            f.write(full_text + "\n\n")

    # ---------- 存到 Firebase（簡短摘要） ----------
    try:
        doc_ref = db.collection(result_field).document(today.strftime("%Y%m%d"))
        doc_ref.set({
            target: short_text
        })
        if not SILENT_MODE:
            print(f"[Firebase] {target} 簡短摘要已存成功")
    except Exception as e:
        print("[Firebase] 存檔失敗:", e)

    return full_text, short_text

# ---------- 主程式 ----------
def main():
    db = get_db()
    targets = [
        ("台積電", NEWS_COLLECTION_TSMC),
        ("鴻海", NEWS_COLLECTION_FOX),
        ("聯電", NEWS_COLLECTION_UMC)
    ]
    result_field = "RESULT_SUMMARY"

    for target_name, collection_name in targets:
        print(f"\n=== 分析 {target_name} ===")
        full_text, short_text = analyze_target(db, collection_name, target_name, result_field)
        if not SILENT_MODE:
            print("\n[完整分析 TXT]")
            print(full_text)
            print("\n[Firebase 簡短摘要]")
            print(short_text)

if __name__ == "__main__":
    main()
