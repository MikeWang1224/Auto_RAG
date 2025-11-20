# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
輸出格式精簡版：取用新聞 + 偏向 + 總分
"""

import os, signal, regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

SILENT_MODE = True
TAIWAN_TZ = timezone(timedelta(hours=8))

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

SENSITIVE_WORDS = {
    "法說": 1.5, "財報": 1.4, "新品": 1.3, "合作": 1.3, "併購": 1.4,
    "投資": 1.3, "停工": 1.6, "下修": 1.5, "利空": 1.5, "爆料": 1.4,
    "營收": 1.3, "展望": 1.2,
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

@dataclass
class Token:
    polarity: str
    ttype: str
    pattern: str
    weight: float
    note: str

def get_db():
    return firestore.Client()

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def parse_docid_time(doc_id: str):
    m = re.match(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$", doc_id or "")
    if not m:
        return None
    ymd, hms = m.group("ymd"), m.group("hms") or "000000"
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
                continue
        else:
            compiled.append(("substr", None, t.weight, t.note, t.pattern.lower()))
    return compiled

def score_text(text: str, pos_c, neg_c, target: str = None) -> float:
    norm = normalize(text)
    score = 0.0
    aliases = {"台積電": ["台積電", "tsmc", "2330"],
               "鴻海": ["鴻海", "foxconn", "2317", "富士康"],
               "聯電": ["聯電", "umc", "2303"]}
    company_pattern = "|".join(re.escape(a) for a in aliases.get(target, []))
    if not re.search(company_pattern, norm):
        return 0.0
    for ttype, cre, w, note, patt in pos_c + neg_c:
        matched = cre.search(norm) if ttype == "regex" else patt in norm
        if matched:
            score += w
    return score

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

# ---------- 分析函式 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()

    filtered = []
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt: continue
        delta_days = (today - dt.date()).days
        if delta_days > 2: continue

        day_weight = 1.0 if delta_days == 0 else 0.85 if delta_days == 1 else 0.7
        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict): continue
            title, content = v.get("title", ""), v.get("content", "")
            full = title + " " + content
            score = score_text(full, pos_c, neg_c, target)
            if score == 0: continue
            adj_score = adjust_score_for_context(full, score)
            filtered.append((d.id, k, title, adj_score * day_weight))

    if filtered:
        filtered.sort(key=lambda x: abs(x[3]), reverse=True)
        top_news = filtered[:10]

        print(f"\n📰 {target} 近期重點新聞（取用）：")
        for docid, key, title, _ in top_news:
            print(f"[{docid}#{key}] {title}")

        avg_score = sum(s for _, _, _, s in top_news) / len(top_news)
        if avg_score >= 2:
            trend = "上漲 🔼"
        elif 0 < avg_score < 2:
            trend = "微漲 ↗️"
        elif -2 < avg_score <= 0:
            trend = "微跌 ↘️"
        elif avg_score <= -2:
            trend = "下跌 🔽"
        else:
            trend = "不明確 ⚖️"

        print(f"\n明日偏向：{trend}")
        print(f"總分：{int(round(avg_score))}\n")

        try:
            db.collection(result_field).document(today.strftime("%Y%m%d")).set({
                "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
                "trend": trend,
                "score": int(round(avg_score)),
                "news_list": [{"docid": d, "key": k, "title": t} for d, k, t, _ in top_news]
            })
        except Exception as e:
            print(f"[warning] Firestore 寫回失敗：{e}")

    else:
        print(f"\n📰 {target} 近期重點新聞：無可用新聞")
        print(f"明日偏向：不明確 ⚖️")
        print(f"總分：0\n")

# ---------- 主程式 ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
