# -*- coding: utf-8 -*-
"""
股票新聞分析工具（修正版：新聞命中時僅取與公司相關的句子）
 - 若新聞中沒有提到該公司，則不納入命中。
 - Groq 分析邏輯不變。
"""

import os
import re
import time
import json
import warnings
from datetime import datetime
from firebase_admin import credentials, firestore, initialize_app


# ========= Firestore 初始化 =========
if not len(firebase_admin._apps):
    cred = credentials.Certificate("gcp-key.json")
    initialize_app(cred)
db = firestore.client()

# ========= 公司關鍵字設定 =========
COMPANY_KEYWORDS = {
    "台積電": ["台積電", "TSMC", "2330"],
    "鴻海": ["鴻海", "Foxconn", "2317"],
    "聯電": ["聯電", "UMC", "2303"],
}

# ========= 分數與命中詞庫設定 =========
SCORE_THRESHOLD = 0.8

POSITIVE_TOKENS = ["大漲", "上漲", "飆升", "創高", "獲利", "突破", "看好", "買盤湧入", "漲停"]
NEGATIVE_TOKENS = ["下跌", "重挫", "走跌", "翻黑", "疲弱", "賣壓", "利空", "跌停", "衰退"]


# ========= 工具函式 =========
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def score_text(text: str, pos_tokens, neg_tokens):
    """
    根據命中詞數打分：正面詞 +1，負面詞 -1
    """
    pos_hits = [t for t in pos_tokens if t in text]
    neg_hits = [t for t in neg_tokens if t in text]
    score = len(pos_hits) - len(neg_hits)
    total = len(pos_hits) + len(neg_hits)
    if total == 0:
        return {"score": 0, "pos_hits": [], "neg_hits": []}
    return {"score": score / total, "pos_hits": pos_hits, "neg_hits": neg_hits}


def extract_company_related_text(text: str, company: str) -> str:
    """
    從文章中提取包含公司關鍵字的句子或片段。
    若找不到，回傳空字串。
    """
    keywords = COMPANY_KEYWORDS.get(company, [company])
    # 改成更寬鬆的分段方式（包含逗號、頓號、空格）
    parts = re.split(r'[。！？；,.、\s]', text)
    related = [p.strip() for p in parts if any(kw in p for kw in keywords)]

    # 若沒有直接命中，嘗試從全文找出包含公司名的片段
    if not related:
        for kw in keywords:
            match = re.search(rf'[^。！？；.!?]*{kw}[^。！？；.!?]*', text)
            if match:
                related.append(match.group().strip())

    return "。".join(related)


# ========= 主邏輯 =========
def analyze_target(target: str, news_list: list):
    """
    分析指定公司的新聞，並回傳命中清單。
    """
    print(f"[info] ===== 開始分析 {target} =====")
    matched_news = []

    for it in news_list:
        title = it.get("title", "")
        content = it.get("content", "")
        text_raw = f"{title}。{content}"

        # 取出與公司相關的句子
        text_for_score = extract_company_related_text(text_raw, target)
        if not text_for_score.strip():
            print(f"[warn] {target} 無關新聞：{title}")
            continue
        else:
            print(f"[ok] {target} 命中句子：{text_for_score}")

        # Token 打分
        res = score_text(text_for_score, POSITIVE_TOKENS, NEGATIVE_TOKENS)

        # 若分數超過門檻，加入命中清單
        if abs(res["score"]) >= SCORE_THRESHOLD:
            matched_news.append({
                "id": it.get("id", ""),
                "title": title,
                "score": res["score"],
                "pos_hits": res["pos_hits"],
                "neg_hits": res["neg_hits"],
                "related_text": text_for_score
            })

    print(f"[info] 過濾後新聞：{len(matched_news)} / {len(news_list)}")
    if not matched_news:
        print(f"[info] 無符合條件的新聞\n")
        return []

    # 只取前 5 則
    matched_news = sorted(matched_news, key=lambda x: -abs(x["score"]))[:5]
    return matched_news


# ========= 測試或整合範例 =========
if __name__ == "__main__":
    # 模擬資料
    news_data = [
        {
            "id": "news_8",
            "title": "台積電上漲25元開高！台股一度飆近400點 鴻海、廣達卻慘翻黑...",
            "content": "台股今日開盤氣勢如虹，台積電領漲，但鴻海、廣達表現疲弱。",
        },
        {
            "id": "news_9",
            "title": "鴻海宣布新電動車計畫，市場反應熱烈！",
            "content": "法人指出鴻海未來成長動能強勁，投資人樂觀看待。",
        },
    ]

    # 分析兩家公司
    tsmc_hits = analyze_target("台積電", news_data)
    honhai_hits = analyze_target("鴻海", news_data)

    print("\n=== 台積電命中結果 ===")
    for h in tsmc_hits:
        print(f"【{h['title']}】→ 分數 {h['score']:.2f}")

    print("\n=== 鴻海命中結果 ===")
    for h in honhai_hits:
        print(f"【{h['title']}】→ 分數 {h['score']:.2f}")
