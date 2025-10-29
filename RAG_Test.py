# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
格式完全比照範例輸出：
- 每則新聞有命中列表與趨勢符號
- 最後用 Groq 總結明日走勢與原因
- Firestore 寫回結果
"""

import os, signal, regex as re, sys, io
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 設定 ----------
SILENT_MODE = False
MAX_DISPLAY_NEWS = 5
TAIWAN_TZ = timezone(timedelta(hours=8))

# ---------- 讀 .env ----------
if os.path.exists(".env"):
    load_dotenv(".env", override=True)

TOKENS_COLLECTION = os.getenv("FIREBASE_TOKENS_COLLECTION", "bull_tokens")
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.2"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "2"))

STOP = False
def _sigint_handler(signum, frame):
    global STOP
    STOP = True
    print("\n[info] 偵測到 Ctrl+C，將安全停止…")
signal.signal(signal.SIGINT, _sigint_handler)

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
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def first_n_sentences(text: str, n: int = 3) -> str:
    parts = re.split(r'(?<=[。\.！!\?？；;])\s*', text.strip())
    parts = [p for p in parts if p.strip()]
    joined = "".join(parts[:n])
    if not re.search(r'[。\.！!\?？；;]$', joined):
        joined += "..."
    return joined

def parse_docid_time(doc_id: str):
    try:
        if "_" in doc_id:
            return datetime.strptime(doc_id, "%Y%m%d_%H%M%S").replace(tzinfo=TAIWAN_TZ)
        return datetime.strptime(doc_id, "%Y%m%d").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

# ---------- 初始化 ----------
def get_db(): return firestore.Client()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Token 處理 ----------
def load_tokens(db):
    pos, neg = [], []
    for d in db.collection(TOKENS_COLLECTION).stream():
        data = d.to_dict() or {}
        t = Token(
            polarity=data.get("polarity",""),
            ttype=data.get("type","substr"),
            pattern=data.get("pattern",""),
            weight=float(data.get("weight",1.0)),
            note=data.get("note","")
        )
        if t.polarity.lower() == "positive":
            pos.append(t)
        elif t.polarity.lower() == "negative":
            neg.append(t)
    return pos, neg

# ---------- 打分 ----------
def score_text(text: str, pos_tokens, neg_tokens) -> MatchResult:
    text_norm = normalize(text)
    score, hits = 0.0, []
    for t in pos_tokens + neg_tokens:
        w = t.weight if t.polarity == "positive" else -abs(t.weight)
        matched = re.search(t.pattern, text_norm, re.I) if t.ttype == "regex" else t.pattern.lower() in text_norm
        if matched:
            hits.append((t.pattern, w, t.note))
            score += w
    return MatchResult(score, hits)

# ---------- Groq 總結 ----------
def groq_analyze(news_list, target):
    text_block = "\n".join([f"{i+1}. {n}" for i,n in enumerate(news_list)])
    prompt = f"""你是一位台股分析師。根據以下{target}相關新聞，請判斷明日{target}股價走勢：
回覆格式：
明天{target}股價走勢：<上漲/下跌/不明確> 🔼🔽⚠️
原因：<40字內簡述>

{text_block}
"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":"你是專業股市分析師，回答簡潔準確。"},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=120,
        )
        ans = resp.choices[0].message.content.strip()
        return re.sub(r"\s+"," ",ans)
    except Exception as e:
        return f"明天{target}股價走勢：不明確 ⚠️\n原因：Groq分析失敗({e})"

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    items = db.collection(collection).stream()
    now = datetime.now(TAIWAN_TZ)
    start = now - timedelta(days=LOOKBACK_DAYS)

    output_lines, groq_inputs, filtered = [], [], []

    for d in items:
        t = parse_docid_time(d.id)
        if not t or t < start: continue
        data = d.to_dict() or {}
        for k,v in data.items():
            if not isinstance(v,dict): continue
            title, content = v.get("title",""), v.get("content","")
            full = title + " " + content
            res = score_text(full, pos, neg)
            if abs(res.score) < SCORE_THRESHOLD or not res.hits:
                continue
            trend = "✅ 明日可能大漲" if res.score > 0 else "❌ 明日可能下跌"
            hit_lines = [f"  {'+' if w>0 else '-'} {p}（{n}）" for p,w,n in res.hits]
            part = f"[{d.id}#{k}]\n標題：{first_n_sentences(title)}\n{trend}\n命中：\n" + "\n".join(hit_lines)
            output_lines.append(part+"\n")
            groq_inputs.append(full)
            filtered.append((d.id, k, res))

    if not filtered:
        return f"{target}：無明顯變化\n"

    groq_result = groq_analyze(groq_inputs, target)
    output = "\n".join(output_lines) + "\n" + groq_result + "\n"

    # 寫回 Firestore
    for doc_id, key, res in filtered:
        try:
            db.collection(collection).document(doc_id).set({
                result_field: {
                    key: {
                        "summary": groq_result,
                        "trend": "上漲" if res.score>0 else "下跌",
                        "reason": groq_result,
                        "hits": [{"pattern":p,"weight":w,"note":n} for p,w,n in res.hits],
                        "updated_at": datetime.now(TAIWAN_TZ).isoformat()
                    }
                }
            }, merge=True)
        except Exception as e:
            print(f"[warning] Firestore 寫回失敗 {doc_id}#{key}: {e}")
    return output

# ---------- 主程式 ----------
def main():
    print("🚀 開始分析台股焦點股...\n")
    db = get_db()
    today = datetime.now(TAIWAN_TZ).strftime("%Y%m%d")
    os.makedirs("results", exist_ok=True)
    result_file = f"results/result_{today}.txt"

    results = []
    for i,(target,col,field) in enumerate([
        ("台積電", NEWS_COLLECTION_TSMC, "Groq_result"),
        ("鴻海", NEWS_COLLECTION_FOX, "Groq_result_Foxxcon"),
        ("聯電", NEWS_COLLECTION_UMC, "Groq_result_UMC"),
    ]):
        print(f"📈 分析：{target}")
        res = analyze_target(db, col, target, field)
        results.append(f"{res.strip()}\n")
        if i < 2:
            print("="*70)

    final_output = "\n" + ("="*70 + "\n").join(results)
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\n✅ 結果已儲存至：{result_file}")

if __name__ == "__main__":
    main()
