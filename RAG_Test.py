import os
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict
from google.cloud import firestore

# ---------- 常數 ----------
TOKENS_COLLECTION = os.getenv("FIREBASE_TOKENS_COLLECTION", "bull_tokens")
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"  # ✅ 聯電
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "3.0"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "2"))
TAIWAN_TZ = timezone(timedelta(hours=8))


# ---------- 資料結構 ----------
@dataclass
class MatchResult:
    score: float
    reasons: List[str]


# ---------- Firestore 初始化 ----------
def get_db():
    """不使用 firebase_admin，直接用 google-cloud-firestore"""
    return firestore.Client()


# ---------- 工具函式 ----------
def parse_docid_time(docid: str):
    try:
        m = re.search(r"(\d{8})[_-](\d{6})", docid)
        if not m:
            return None
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except Exception:
        return None


# ---------- 改良後新聞載入 ----------
def load_news_items(db, col_name: str, days: int) -> List[Dict]:
    items, seen = [], set()
    now = datetime.now(TAIWAN_TZ)
    start = now - timedelta(days=days)

    print(f"[debug] 🔍 正在讀取 Firestore 集合：{col_name}")
    for d in db.collection(col_name).stream():
        data = d.to_dict() or {}
        print(f"[debug] → 文件 {d.id} keys={list(data.keys())}")
        dt = parse_docid_time(d.id)
        if dt and dt < start:
            continue

        found = False
        for k, v in data.items():
            if k.startswith("news_") and isinstance(v, dict):
                title, content = str(v.get("title") or ""), str(v.get("content") or "")
                if title or content:
                    uniq = f"{title}|{content}"
                    if uniq not in seen:
                        seen.add(uniq)
                        items.append({"id": f"{d.id}#{k}", "title": title, "content": content, "ts": dt})
                        found = True

        if not found and ("title" in data or "content" in data):
            title, content = str(data.get("title") or ""), str(data.get("content") or "")
            uniq = f"{title}|{content}"
            if uniq not in seen and (title or content):
                seen.add(uniq)
                items.append({"id": d.id, "title": title, "content": content, "ts": dt})

    print(f"[debug] ✅ 共載入 {len(items)} 篇新聞\n")
    items.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=TAIWAN_TZ), reverse=True)
    return items


# ---------- 關鍵字分析 ----------
def score_text(text: str, target: str) -> MatchResult:
    aliases = {
        "台積電": ["台積電", "tsmc", "2330"],
        "鴻海": ["鴻海", "hon hai", "2317", "foxconn", "富士康"],
        "聯電": ["聯電", "umc", "2303"],
    }

    text_norm = text.lower()
    alias_pattern = "|".join(map(re.escape, aliases.get(target, [target])))

    # 聯電：允許相關詞命中
    related_umc_keywords = ["晶圓代工", "成熟製程", "8吋", "8 吋", "車用晶片", "驅動ic", "中階製程", "代工廠"]

    if target == "聯電":
        alias_or_related_pattern = alias_pattern + "|" + "|".join(map(re.escape, related_umc_keywords))
        if not re.search(alias_or_related_pattern, text_norm):
            return MatchResult(0.0, [])
    else:
        if not re.search(alias_pattern, text_norm):
            return MatchResult(0.0, [])

    score = 4.0
    reasons = [f"命中關鍵詞，與 {target} 相關"]
    return MatchResult(score, reasons)


# ---------- 主分析 ----------
def analyze_target(db, collection_name: str, target: str):
    print(f"🔎 開始分析 {target} ({collection_name}) ...")
    items = load_news_items(db, collection_name, LOOKBACK_DAYS)
    if not items:
        print(f"⚠️ 未找到任何 {target} 的新聞。")
        return

    hits = 0
    for item in items:
        text = f"{item['title']} {item['content']}"
        result = score_text(text, target)
        if result.score >= SCORE_THRESHOLD:
            hits += 1
            print(f"[HIT] {item['title']} ({result.score:.1f}) -> {result.reasons}")

    if hits == 0:
        print(f"📭 沒有符合條件的 {target} 新聞。")
    else:
        print(f"✅ 共 {hits} 篇與 {target} 相關的新聞。")


# ---------- 主程式 ----------
def main():
    db = get_db()

    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電")
    print("\n" + "=" * 70 + "\n")
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海")
    print("\n" + "=" * 70 + "\n")
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電")  # ✅ 聯電


if __name__ == "__main__":
    main()
