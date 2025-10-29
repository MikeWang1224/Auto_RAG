# -*- coding: utf-8 -*-
"""
股票新聞分析（多公司版）
✅ 輸出顯示在終端
✅ 同步寫回 Firebase
✅ 自動儲存結果於 results/ 下
"""

import os
import signal
import regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from google.cloud import firestore
from dotenv import load_dotenv
from groq import Groq

# ---------- 全域設定 ----------
SILENT_MODE = False
MAX_DISPLAY_NEWS = 5
TAIWAN_TZ = timezone(timedelta(hours=8))
STOP = False

# ---------- 讀取環境變數 ----------
if os.path.exists(".env"):
    load_dotenv(".env", override=True)

PROJECT_ID = os.getenv("FIREBASE_PROJECT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"
SCORE_THRESHOLD = 0.5
LOOKBACK_DAYS = 2

# ---------- Ctrl+C 安全停止 ----------
def _sigint_handler(signum, frame):
    global STOP
    STOP = True
    print("\n⚠️ 偵測到 Ctrl+C，停止中…")
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
    hits: list[tuple[str, float, str]]

# ---------- 工具 ----------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def first_n_sentences(text: str, n: int = 3) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[。\.！!\?？；;])\s*', text.strip())
    parts = [p for p in parts if p.strip()]
    joined = "".join(parts[:n])
    if not re.search(r'[。\.！!\?？；;]$', joined):
        joined += "。"
    return joined

# ---------- 初始化 ----------
def init_firestore():
    return firestore.Client(project=PROJECT_ID)

def init_groq():
    return Groq(api_key=GROQ_API_KEY)

# ---------- 載入 Token ----------
def load_tokens(db, collection: str):
    tokens = []
    for d in db.collection(collection).stream():
        t = d.to_dict()
        tokens.append(Token(
            polarity=t.get("polarity", ""),
            ttype=t.get("type", ""),
            pattern=t.get("pattern", ""),
            weight=float(t.get("weight", 1.0)),
            note=t.get("note", "")
        ))
    return tokens

# ---------- 評分 ----------
def score_text(text: str, pos_tokens, neg_tokens, target: str):
    text_norm = normalize(text)
    total = 0.0
    hits = []
    for tok in pos_tokens + neg_tokens:
        found = False
        if tok.ttype == "substr" and tok.pattern in text_norm:
            found = True
        elif tok.ttype == "regex" and re.search(tok.pattern, text_norm):
            found = True
        if found:
            w = tok.weight if tok.polarity == "positive" else -tok.weight
            total += w
            hits.append((tok.pattern, w, tok.note))
    return MatchResult(score=total, hits=hits)

# ---------- Firestore 寫入 ----------
def write_result(db, collection, doc_id, data):
    ref = db.collection(collection).document(doc_id)
    ref.set(data, merge=True)

# ---------- 主分析函數 ----------
def analyze_target(db, news_collection, target_name, result_collection, force_dir=False):
    pos_tokens = load_tokens(db, TOKENS_COLLECTION)
    neg_tokens = [t for t in pos_tokens if t.polarity == "negative"]
    pos_tokens = [t for t in pos_tokens if t.polarity == "positive"]

    now = datetime.now(TAIWAN_TZ)
    since = now - timedelta(days=LOOKBACK_DAYS)
    news_docs = list(db.collection(news_collection).stream())

    terminal_logs = []
    for doc in news_docs:
        if STOP:
            break
        it = doc.to_dict()
        it["id"] = doc.id
        text = it.get("content") or it.get("title") or ""
        res = score_text(text, pos_tokens, neg_tokens, target_name)
        if abs(res.score) >= SCORE_THRESHOLD and res.hits:
            trend = "✅ 明日可能大漲" if res.score > 0 else "❌ 明日可能下跌"
            hits_text_lines = [
                f"  {'+' if w>0 else '-'} {patt}（{note}）" for patt, w, note in res.hits
            ]
            truncated_title = first_n_sentences(it.get("title", ""), 3)
            terminal_logs.append(
                f"""[{it['id']}]
標題：{truncated_title}
{trend}
命中：
""" + "\n".join(hits_text_lines) + "\n"
            )

    # 輸出結果文字
    if not terminal_logs:
        result_text = f"{target_name}：無明顯變化"
    else:
        result_text = "\n".join(terminal_logs)
    print(result_text)

    # 寫回 Firestore
    write_result(db, result_collection, now.strftime("%Y%m%d"), {
        "summary": result_text,
        "updated": now.isoformat(),
    })

    return result_text

# ---------- 主程式 ----------
def main():
    os.makedirs("results", exist_ok=True)
    db = init_firestore()
    now = datetime.now(TAIWAN_TZ)

    print("🚀 開始分析台股焦點股...\n")

    all_results = []
    targets = [
        ("台積電", NEWS_COLLECTION_TSMC, "Groq_result", False),
        ("鴻海", NEWS_COLLECTION_FOX, "Groq_result_Foxxcon", True),
        ("聯電", NEWS_COLLECTION_UMC, "Groq_result_UMC", True),
    ]

    for i, (target, col, result_col, force_dir) in enumerate(targets):
        print(f"📈 分析：{target}")
        result_text = analyze_target(db, col, target, result_col, force_dir)
        all_results.append(result_text)
        if i < len(targets) - 1:
            print("=" * 70)

    # 儲存文字檔
    file_path = f"results/result_{now.strftime('%Y%m%d')}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_results))
    print(f"\n✅ 結果已儲存至：{file_path}")

# ---------- 執行 ----------
if __name__ == "__main__":
    main()
