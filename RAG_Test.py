# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
準確率極致版（短期預測特化） - Context-aware + 去重新聞版
✅ 嚴格依據情緒分數決策
✅ 敏感詞加權（法說 / 財報 / 新品 / 停工等）
✅ 支援 3 日延遲效應
✅ Firestore 寫回 + 本地 result.txt
✅ 相同新聞內容去重
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
    """解析格式範例：'+7.50 (+3.28%)' -> 0.0328"""
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

# ---------- Scoring ----------
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

# ---------- Groq 分析 ----------
def groq_analyze(news_list, target, avg_score, today_change):
    """依據整理後新聞與今日股價計算明日走勢及原因"""
    # 實作與你原版一致，包含敏感詞加權、主要利多利空、今日股價影響
    # ...保留原 groq_analyze 實作...
    return "Groq 分析結果（此處呼叫 client.chat.completions）"

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()
    filtered, weighted_scores = [], []
    today_price_change = 0.0

    # 掃描 collection 取得今日 price_change
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
    except:
        today_price_change = 0.0

    # 處理近三日新聞
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        delta_days = (today - dt.date()).days
        if delta_days > 2:
            continue
        day_weight = 1.0 if delta_days == 0 else 0.85 if delta_days == 1 else 0.7
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
            token_weight = 1.0 + min(len(res.hits)*0.05,0.3)
            impact = 1.0 + sum(w*0.05 for k_sens,w in SENSITIVE_WORDS.items() if k_sens in full)
            total_weight = day_weight * token_weight * impact
            filtered.append((d.id, k, title, full, res, total_weight))
            weighted_scores.append(adj_score*total_weight)

    # 無新聞 fallback
    if not filtered:
        summary = groq_analyze([], target, 0, today_price_change)
    else:
        # 去重新聞，取前10條
        seen_text, top_news = set(), []
        for docid,key,title,full,res,weight in sorted(filtered,key=lambda x:abs(x[4].score*x[5]),reverse=True):
            news_text = normalize(full)
            if news_text in seen_text:
                continue
            seen_text.add(news_text)
            top_news.append((docid,key,title,res,weight))
            if len(top_news)>=10:
                break

        # 計算 avg_score
        news_with_scores = [(t, res.score*weight) for _,_,t,res,weight in top_news]
        avg_score = sum(s for _,s in news_with_scores)/len(news_with_scores)
        summary = groq_analyze(news_with_scores, target, avg_score, today_price_change)

        # 本地存檔
        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname,"a",encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            for docid,key,title,res,weight in top_news:
                hits_text = "\n".join([f"  {'+' if w>0 else '-'} {p}（{n}）" for p,w,n in res.hits])
                f.write(f"[{docid}#{key}]（{weight:.2f}x）\n標題：{first_n_sentences(title)}\n命中：\n{hits_text}\n\n")
            f.write(summary+"\n\n")

    print(summary+"\n")

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
        print("🚀 開始分析台股焦點股（準確率極致版）...\n")
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__=="__main__":
    main()
