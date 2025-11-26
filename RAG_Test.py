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

load_dotenv()
FIREBASE_CREDENTIAL_JSON = os.getenv("FIREBASE_CREDENTIAL_JSON", "serviceAccount.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = FIREBASE_CREDENTIAL_JSON

SENSITIVE_WORDS = {
    "官司": 2,
    "調查": 2,
    "裁員": 3,
    "虧損": 3,
    "財報不佳": 3,
    "爆炸": 3,
    "災情": 3,
    "倒閉": 4,
}

NEWS_COLLECTION_TSMC = "Tw_TSMC_News_Collection"
NEWS_COLLECTION_FOX = "Tw_Foxconn_News_Collection"
NEWS_COLLECTION_UMC = "Tw_UMC_News_Collection"

# ---------- Firestore ----------
def get_db():
    return firestore.Client()

# ---------- dataclass ----------
@dataclass
class ScoreResult:
    score: float
    hits: List[Tuple[str, int, int]]

# ---------- Token Loader ----------
def load_tokens(db):
    pos, neg = [], []
    try:
        for d in db.collection("positive_tokens").stream():
            pos.append(d.to_dict().get("token"))
    except: pass
    try:
        for d in db.collection("negative_tokens").stream():
            neg.append(d.to_dict().get("token"))
    except: pass
    return pos, neg

def compile_tokens(tokens):
    return [re.compile(re.escape(t), re.IGNORECASE) for t in tokens]

# ---------- 分析工具 ----------
def normalize(txt): return re.sub(r"\s+", " ", txt.strip())
def parse_docid_time(s):
    try: return datetime.strptime(s, "%Y%m%d_%H%M%S")
    except: return None

def parse_price_change(v):
    try:
        return float(v.replace("%","")) / 100 if isinstance(v,str) else float(v)
    except: 
        return 0.0

# ---------- Score 主邏輯 ----------
def score_text(full, pos_c, neg_c, target):
    score, hits = 0, []
    for p in pos_c:
        m = list(p.finditer(full))
        if m:
            score += len(m)
            hits.append((p.pattern, +1, len(m)))
    for n in neg_c:
        m = list(n.finditer(full))
        if m:
            score -= len(m)
            hits.append((n.pattern, -1, len(m)))
    return ScoreResult(score, hits)

def adjust_score_for_context(text, score):
    if "調查" in text and "澄清" in text: score += 1
    if "謠言" in text and score < 0: score += 1
    return score

def detect_divergence(avg_score, change):
    if avg_score > 1 and change < -0.02:
        return "正面新聞但股價下跌（可能提前反應或獲利了結）"
    if avg_score < -1 and change > 0.02:
        return "負面新聞但股價上漲（可能強勢反轉）"
    return "無明顯背離"

def adjust_by_market(avg_score, change):
    if change < -0.02: return avg_score * 1.05
    if change > 0.02: return avg_score * 0.95
    return avg_score

# ---------- 硬規則 ----------
def decide_by_hard_rules(news_list, today_change, full_texts, adjusted_avg=None, divergence=None):
    total = sum(s for _, s in news_list) if news_list else 0
    if total < -1: direction = "下跌 🔽"
    elif total > 1: direction = "上漲 🔼"
    else: direction = "持平 ⬜"

    reason = "、".join([t for t, s in sorted(news_list, key=lambda x: abs(x[1]), reverse=True)[:3]]) if news_list else "缺乏重大新聞"

    return f"明天股價走勢：{direction} 原因：{reason}。 情緒分數：{round(total,2)}"

# ---------- 替代 Groq ----------
def groq_analyze(news_list, target, avg_score, today_change, adjusted_avg=None, divergence=None):
    res = decide_by_hard_rules(news_list, today_change, [], adjusted_avg, divergence)
    return res.replace("明天股價走勢", f"明天{target}股價走勢", 1)

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()

    filtered, weighted_scores = [], []
    today_price_change = 0.0

    # 讀今日漲跌
    try:
        for d in db.collection(collection).stream():
            dt = parse_docid_time(d.id)
            if not dt or dt.date() != today: continue
            for _, v in (d.to_dict() or {}).items():
                if isinstance(v,dict) and "price_change" in v:
                    today_price_change = parse_price_change(v["price_change"])
                    break
    except: pass

    # 掃新聞
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt: continue
        delta = (today - dt.date()).days
        if delta > 2: continue

        day_weight = 1.0 if delta == 0 else 0.85 if delta == 1 else 0.7
        data = d.to_dict() or {}

        for k,v in data.items():
            if not isinstance(v,dict): continue
            title, content = v.get("title",""), v.get("content","")
            full = title + " " + content

            res = score_text(full, pos_c, neg_c, target)
            if not res.hits: continue

            adj = adjust_score_for_context(full, res.score)
            token_w = 1.0 + min(len(res.hits)*0.05, 0.3)
            impact = 1.0 + sum(w*0.05 for k_s,w in SENSITIVE_WORDS.items() if k_s in full)

            weight = day_weight * token_w * impact

            filtered.append((d.id, k, title, full, res, weight))
            weighted_scores.append(adj * weight)

    # fallback（無新聞）
    if not filtered:
        summary = groq_analyze([], target, 0, today_price_change)
    else:
        seen, top_news = set(), []
        for docid, key, title, full, res, w in sorted(filtered, key=lambda x: abs(x[4].score*x[5]), reverse=True):
            norm = normalize(full)
            if norm in seen: continue
            seen.add(norm)
            top_news.append((docid,key,title,res,w,full))
            if len(top_news)>=10: break

        news_with_scores = [(t, res.score * w) for _,_,t,res,w,_ in top_news]
        avg_score = sum(s for _,s in news_with_scores) / len(news_with_scores)

        adjusted_avg = adjust_by_market(avg_score, today_price_change)
        divergence = detect_divergence(avg_score, today_price_change)
        summary = groq_analyze(news_with_scores, target, avg_score, today_price_change, adjusted_avg, divergence)

    print(summary + "\n")

    # ---------- ★★★ 修改後：Firebase 只寫入 summary（精簡版） ★★★
    try:
        db.collection(result_field).document(today.strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "result": summary,   # 只存這一行！
        })
    except Exception as e:
        print(f"[warning] Firestore 寫回失敗：{e}")

# ---------- 主程式 ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股（硬規則 + 市場調整）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
