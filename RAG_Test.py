# -*- coding: utf-8 -*-
"""
股票新聞分析工具（GitHub Actions 優化版 + 每則新聞詳細輸出）
🔥 TXT = 每則新聞都分析
🔥 Firestore = Groq 直接輸出 3 行短版（固定格式）
"""

import os
import signal
import regex as re
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

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

DOCID_RE = re.compile(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$")

def get_db() -> firestore.Client:
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

# ---------- Token 載入 ----------
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

# ---------- 計分 ----------
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

# ---------- Groq 短版 3 行 ----------
def groq_analyze_batch(news_with_scores: List[Tuple[str, float]], target: str, price_change: str = "") -> str:
    symbol = {"上漲": "🔼","下跌": "🔽","不明確":"⚖️","微漲":"↗️","微跌":"↘️"}
    if not news_with_scores:
        return f"""明天{target}股價走勢：不明確 ⚖️
原因：近三日無相關新聞。今日漲跌：{price_change}
情緒分數：0"""
    combined = "\n".join(f"{i+1}. ({s:+.2f}) {t}" for i, (t, s) in enumerate(news_with_scores))
    avg_score = sum(s for _, s in news_with_scores)/len(news_with_scores)
    prompt_text = f"""
你是一位專業台股分析師，請依以下規則輸出答案：
⚠️ 必須嚴格輸出三行：
明天{target}股價走勢：{{上漲／下跌／不明確}} {{符號}}
原因：一句原因
情緒分數：-10~10

平均情緒分數：{avg_score:+.2f}
{combined}
"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":"你是專業台股分析師。必須輸出三行短版。"},
                      {"role":"user","content":prompt_text}],
            temperature=0.1,
            max_tokens=150,
            timeout=25
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"""明天{target}股價走勢：不明確 ⚖️
原因：Groq 分析失敗({e})
情緒分數：0"""

# ---------- TXT 詳細輸出 ----------
def dump_detailed_news(target: str, today, all_news: List[Tuple]):
    fname = os.path.join(RESULTS_DIR, f"result_{today.strftime('%Y%m%d')}.txt")
    with open(fname, "a", encoding="utf-8") as f:
        f.write(f"📰 {target} 近期新聞詳細分析（含衝擊）:\n\n")
        for docid, key, title, res, weight in all_news:
            f.write(f"[{docid}#{key}] ({weight:.2f}x, 分數={res.score:+.2f}) {first_n_sentences(title)}\n")
            for patt, w, note in res.hits:
                sign = "+" if w>0 else "-"
                f.write(f"   {sign} {patt}（{note}）\n")
            f.write("\n")

# ---------- 主分析 ----------
def analyze_target(db, collection: str, target: str, result_field: str):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)

    today = datetime.now(TAIWAN_TZ).date()
    all_news = []
    price_change = ""

    # Firestore 拉取
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        delta_days = (today - dt.date()).days
        if delta_days > 2:
            continue
        day_weight = {0:1.0,1:0.85,2:0.7}.get(delta_days,0.7)
        data = d.to_dict() or {}
        for k,v in data.items():
            if not isinstance(v, dict):
                continue
            if not price_change:
                price_change = v.get("price_change","")
            title, content = v.get("title",""), v.get("content","")
            res = score_text(title + " " + content, pos_c, neg_c, target)
            if not res.hits:
                continue
            token_weight = 1.0 + min(len(res.hits)*0.05,0.3)
            total_weight = day_weight * token_weight
            all_news.append((d.id, k, title, res, total_weight))

    # TXT 輸出
    if all_news:
        dump_detailed_news(target, today, all_news)

    # Firestore 短版
    if not all_news:
        summary = groq_analyze_batch([], target, price_change)
    else:
        all_news_sorted = sorted(all_news, key=lambda x: abs(x[3].score*x[4]), reverse=True)
        news_with_scores = [(t,res.score*weight) for _,_,t,res,weight in all_news_sorted[:10]]
        summary = groq_analyze_batch(news_with_scores, target, price_change)

        # 同步寫入 TXT
        fname = os.path.join(RESULTS_DIR, f"result_{today.strftime('%Y%m%d')}.txt")
        with open(fname,"a",encoding="utf-8") as f:
            f.write(summary + "\n\n")

    print(summary + "\n")

    # Firestore 寫回
    try:
        db.collection(result_field).document(today.strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "result": summary,
        })
    except Exception as e:
        print("[warning] Firestore 寫回失敗：", e)

# ---------- main ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股...\n")
    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
