# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
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

# 內部敏感詞表（舊版保留）
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

# 硬規則加權
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

# ---------- 市場調整與背離 ----------
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

# ---------- 硬規則決策 fallback ----------
def decide_by_hard_rules(news_list: List[Tuple[str, float]], today_change: float) -> str:
    if not news_list:
        return "明天股價走勢：不明確 ⚖️\n情緒分數：0"
    total_score = sum(1.0 if s>0 else -1.0 if s<0 else 0 for _, s in news_list)
    standardized = total_score / (len(news_list)+1)
    if standardized >= 1.0:
        trend, symbol = "微漲", "↗️"
    elif standardized > -1.0:
        trend, symbol = "微跌", "↘️"
    else:
        trend, symbol = "下跌", "🔽"
    mood_score = max(-10, min(10, int(round(standardized*3))))
    return f"明天股價走勢：{trend} {symbol}\n情緒分數：{mood_score:+d}"

# ---------- Groq LLM 分析 ----------
def groq_llm_analyze(news_list: List[Tuple[str, float]], target: str, today_change: float) -> Tuple[str, str]:
    if not news_list:
        fallback_result = f"明天{target}股價走勢：不明確 ⚖️\n情緒分數：0"
        return fallback_result, "近三日無相關新聞，依市場資訊推算"

    prompt = f"你是一個股票新聞分析專家。\n個股：{target}\n今日股價漲跌幅：{today_change*100:.2f}%\n最近新聞標題及分數：\n"
    for i, (title, score) in enumerate(news_list):
        prompt += f"{i+1}. {title} (score: {score:+.2f})\n"
    prompt += "\n請生成明天股價走勢與情緒分數，並簡述原因（40字內）"

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        groq_result_lines = []
        reason_short = ""
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "原因" in line:
                reason_short = line.split("原因")[-1].strip()[:40]
            else:
                groq_result_lines.append(line)
        groq_result = "\n".join(groq_result_lines)
        if not reason_short:
            reason_short = "依新聞情緒與市場調整推算明日方向"
        return groq_result, reason_short
    except Exception as e:
        print(f"[warning] Groq LLM 呼叫失敗：{e}")
        fallback_result = decide_by_hard_rules(news_list, today_change)
        return fallback_result, "依新聞情緒與市場調整推算明日方向"

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()
    filtered, weighted_scores = [], []
    today_price_change = 0.0

    # 先掃一次 collection 取得今日 price_change
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

    # 新聞打分流程
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
            token_weight = 1.0 + min(len(res.hits)*0.05, 0.3)
            impact = 1.0 + sum(w*0.05 for k_sens, w in SENSITIVE_WORDS.items() if k_sens in full)
            total_weight = day_weight * token_weight * impact
            filtered.append((d.id, k, title, full, res, total_weight))
            weighted_scores.append(adj_score * total_weight)

    # 去重新聞 & top 10
    seen_text = set()
    top_news = []
    for docid, key, title, full, res, weight in sorted(filtered, key=lambda x: abs(x[4].score*x[5]), reverse=True):
        news_text = normalize(full)
        if news_text in seen_text:
            continue
        seen_text.add(news_text)
        top_news.append((docid, key, title, res, weight, full))
        if len(top_news) >= 10:
            break

    # 構造 news_with_scores
    news_with_scores = [(title, res.score*weight) for _, _, title, res, weight, _ in top_news]

    # 使用 Groq LLM
    groq_result, reason_short = groq_llm_analyze(news_with_scores, target, today_price_change)

    # Firestore 寫回
    try:
        db.collection(result_field).document(today.strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "groq_result": groq_result,
            "reason_short": reason_short,
        })
    except Exception as e:
        print(f"[warning] Firestore 寫回失敗：{e}")

    print(f"\n{groq_result}\n原因簡短：{reason_short}\n")

# ---------- 主程式 ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股（Groq LLM + 市場調整）...\n")
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
