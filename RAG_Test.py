# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
整合 price_change 與情緒分數，一次傳給 Groq 做明日股價預測
"""

import os, signal, regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 設定 ----------
SILENT_MODE = False
TAIWAN_TZ = timezone(timedelta(hours=8))

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

STOP = False
def _sigint_handler(signum, frame):
    global STOP
    STOP = True
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

# ---------- Groq 分析 ----------
def groq_analyze(news_list: List[Tuple[str,str,float]], target: str) -> str:
    if not news_list:
        return f"明天{target}股價走勢：不明確 ⚖️\n原因：近三日無相關新聞\n情緒分數：0"

    avg_score = sum(score for _, _, score in news_list) / len(news_list)
    combined = "\n".join(
        f"{i+1}. 標題：{title}\n   當日股價漲跌：{pc}\n   情緒分數：{score:+.2f}"
        for i, (title, pc, score) in enumerate(news_list)
    )

    prompt = f"""
你是一位專業的台股金融分析師，請根據以下「{target}」近三日新聞摘要，
依情緒分數與當日股價漲跌，嚴格推論明日股價方向。

整體平均情緒分數：{avg_score:+.2f}

{combined}

請給出明天股價走勢、原因及情緒分數（-10~+10）。
注意：原因文字必須與股價走勢一致。
"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是台股量化分析員，需依情緒分數與股價漲跌規則產生結論。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=220,
        )
        ans = re.sub(r"\s+", " ", resp.choices[0].message.content.strip())

        m_trend = re.search(r"(上漲|微漲|微跌|下跌|不明確)", ans)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲": "🔼", "微漲": "↗️", "微跌": "↘️", "下跌": "🔽", "不明確": "⚖️"}

        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+?)(?:情緒分數|$)", ans)
        reason = m_reason.group(1).strip() if m_reason else "新聞訊息與股價趨勢整合分析得出的結論。"

        m_score = re.search(r"情緒分數[:：]?\s*(-?\d+)", ans)
        mood_score = int(m_score.group(1)) if m_score else max(-10, min(10, int(round(avg_score * 3))))

        return f"明天{target}股價走勢：{trend} {symbol_map.get(trend,'')}\n原因：{reason}\n情緒分數：{mood_score:+d}"

    except Exception as e:
        return f"明天{target}股價走勢：持平 ⚖️\n原因：Groq分析失敗({e})\n情緒分數：0"

# ---------- 分析流程 ----------
def analyze_target(db, collection_name, target):
    pos_tokens, neg_tokens = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos_tokens), compile_tokens(neg_tokens)

    news_docs = list(db.collection(collection_name)
                     .order_by("timestamp", direction=firestore.Query.DESCENDING)
                     .limit(3).stream())

    news_list = []
    for doc in news_docs:
        data = doc.to_dict()
        title = data.get("title", "")
        price_change = data.get("price_change", "未提供")
        score = score_text(title, pos_c, neg_c, target).score
        score = adjust_score_for_context(title, score)
        news_list.append((title, price_change, score))

    print(groq_analyze(news_list, target))

# ---------- 主程式（舊版風格） ----------
def main():
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電")

if __name__ == "__main__":
    main()
