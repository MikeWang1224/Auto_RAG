# -*- coding: utf-8 -*-
"""
RAG 強化版：新聞 embedding + 查詢 LLM 推論（TSMC / HonHai / UMC）
版本：v1.0
"""

import os
import json
import time
import requests
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore

from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Firebase 初始化
# -----------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# Groq API
# -----------------------------
GROQ_KEY = "你的GROQ_KEY"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 參數
# -----------------------------
COMPANY_KEYWORDS = {
    "TSMC": ["台積電", "2330", "TSMC", "晶圓代工"],
    "HonHai": ["鴻海", "2317", "Foxconn"],
    "UMC": ["聯電", "2303", "UMC"],
}

COLLECTION_NAME = "market_news"

# -----------------------------
# 1. 斷句 & 清洗
# -----------------------------
def clean_text(t):
    if not t:
        return ""
    t = t.replace("\n", " ").replace("\r", " ")
    t = t.replace("（", "(").replace("）", ")")
    return t.strip()

# -----------------------------
# 2. 儲存 embedding 到 Firebase
# -----------------------------
def save_news_with_embedding(date_str, company, news_list):
    doc = db.collection(COLLECTION_NAME).document(date_str)
    exists = doc.get().to_dict() or {}

    if company not in exists:
        exists[company] = []

    for news in news_list:
        text = clean_text(news["title"] + " " + news["content"])
        emb = model_embed.encode(text).tolist()

        exists[company].append({
            "title": news["title"],
            "content": news["content"],
            "embedding": emb,
            "timestamp": datetime.utcnow().isoformat(),
        })

    doc.set(exists)
    print(f"🔥 已寫入 Firebase：{company} 共 {len(news_list)} 則")

# -----------------------------
# 3. 去除重複新聞
# -----------------------------
def dedup_news(news_list):
    cleaned = []
    for item in news_list:
        duplicate = False
        for c in cleaned:
            sim = util.cos_sim(
                model_embed.encode(item["title"]),
                model_embed.encode(c["title"])
            ).item()
            if sim > 0.92:
                duplicate = True
                break
        if not duplicate:
            cleaned.append(item)
    return cleaned

# -----------------------------
# 4. RAG 查詢：找最相似新聞
# -----------------------------
def rag_query(company, date_str):
    doc = db.collection(COLLECTION_NAME).document(date_str).get()
    data = doc.to_dict() or {}

    if company not in data:
        return "無新聞"

    news_items = data[company]

    # RAG 問句
    query = f"{company} 今日股價相關新聞總結 市場情緒？漲跌風險？三點重點？"

    q_emb = model_embed.encode(query)

    # 找相似度最高的前 N 條
    scored = []
    for n in news_items:
        score = util.cos_sim(q_emb, n["embedding"]).item()
        scored.append((score, n))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_news = scored[:5]

    # 整理 context
    context_blocks = []
    for score, item in top_news:
        context_blocks.append(f"[相關度 {round(score,3)}] {item['title']}\n{item['content']}")

    context = "\n\n".join(context_blocks)

    # -----------------------------
    # Groq LLM 回答
    # -----------------------------
    prompt = f"""
你是一位專業的台股分析師。

以下是與 {company} 相關度最高的新聞摘要（RAG 節錄）：

{context}

請用 **極度精準、不可胡亂推測** 的方式回答：

1. 今日整體新聞情緒（正向 / 中立 / 負向）
2. 明日股價「偏漲 + / 偏跌 - / 持平 0」
3. 最關鍵的三則理由

請用以下 JSON 回覆：
{{
  "sentiment": "",
  "prediction": "",
  "reasons": []
}}
"""

    res = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={
            "model": "llama-3.1-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt},
            ]
        }
    ).json()

    msg = res["choices"][0]["message"]["content"]
    return msg

# -----------------------------
# 5. 主流程
# -----------------------------
def main():
    date_str = datetime.now().strftime("%Y-%m-%d")

    # 假設你的新聞抓取結果如下格式
    sample_news = {
        "TSMC": [
            {"title": "台積電法說展望樂觀", "content": "先進製程需求強勁，供應鏈信心提升。"},
            {"title": "外資看好 AI 需求", "content": "帶動台積電長期營運成長。"},
        ],
        "HonHai": [
            {"title": "鴻海電動車專案進度曝光", "content": "新平台開發順利。"},
        ],
        "UMC": [
            {"title": "聯電成熟製程產能改善", "content": "出貨量較上季成長。"},
        ],
    }

    for company in sample_news:
        cleaned = dedup_news(sample_news[company])
        save_news_with_embedding(date_str, company, cleaned)

    # 3家公司 RAG 查詢
    for company in ["TSMC", "HonHai", "UMC"]:
        result = rag_query(company, date_str)
        print("==============")
        print(f"📌 {company} RAG 推論\n{result}")

if __name__ == "__main__":
    main()
