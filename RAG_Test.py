# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from groq import Groq

# ------------------ 初始化 Firebase ------------------ #
key_dict = json.loads(os.environ["NEWS"])
cred = credentials.Certificate(key_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------ 初始化 Groq ------------------ #
groq_api_key = os.environ["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)

# ------------------ Groq 分析 ------------------ #
def analyze_text_with_groq(text):
    prompt = f"""
你是一位股票新聞分析師。以下是關於聯電的最新新聞：
{text}

請你綜合判斷，明天聯電股價的傾向為：
- 上漲
- 下跌
- 持平

請只輸出最有可能的一種情況。
"""
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "你是一位專業的金融分析師"},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()

# ------------------ Firestore 分析與儲存 ------------------ #
def analyze_firestore_news(collection_in, collection_out, stock_name):
    print(f"\n📊 分析 {stock_name} 新聞...")

    docs = list(db.collection(collection_in).stream())
    if not docs:
        print(f"⚠️ 找不到 {collection_in} 資料")
        return

    # 取最新一份新聞文件
    latest_doc = sorted(docs, key=lambda d: d.id, reverse=True)[0]
    news_data = latest_doc.to_dict()
    all_text = "\n\n".join([item["title"] + "\n" + item["content"] for item in news_data.values()])

    print(f"📰 共彙整 {len(news_data)} 則新聞進行分析...")

    result = analyze_text_with_groq(all_text)
    timestamp = datetime.now().strftime("%Y%m%d")

    db.collection(collection_out).document(timestamp).set({
        "result": result,
        "source_doc": latest_doc.id,
        "stock": stock_name,
        "analyzed_at": datetime.now().isoformat()
    })

    print(f"✅ 已儲存 {stock_name} 分析結果至 {collection_out}/{timestamp}")
    print(f"📈 結論：{result}")

# ------------------ 主程式 ------------------ #
if __name__ == "__main__":
    # 台積電（一般分析）
    analyze_firestore_news("NEWS", "Groq_result", "台積電")

    # 鴻海（強制方向分析）
    analyze_firestore_news("NEWS_Foxxcon", "Groq_result_Foxxcon", "鴻海")

    # 聯電（同鴻海邏輯）
    analyze_firestore_news("NEWS_UMC", "Groq_result_UMC", "聯電")
