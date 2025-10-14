# -*- coding: utf-8 -*-
"""
股票新聞分析工具（完整 RAG + Groq 簡潔分析 + Firestore 回傳結果）
版本：台積電 + 鴻海 + 聯電
"""

import os
import time
import json
import warnings
from datetime import datetime
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import firebase_admin
from firebase_admin import credentials, firestore
import requests

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ------------------------------------------------------------
# 🔧 Firestore 初始化
# ------------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate("gcp-key.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

# ------------------------------------------------------------
# 🔍 Groq API 分析邏輯（文字傾向分析）
# ------------------------------------------------------------
def analyze_with_groq(news_list, company_name):
    """呼叫 Groq API 分析新聞方向"""
    if not news_list:
        return f"⚠️ {company_name} 無可分析的新聞"

    joined_text = "\n".join(news_list)
    prompt = f"""
你是一個財經分析AI，請根據以下新聞內容，判斷對「{company_name}」股價的未來一天走勢方向。
請只輸出一行結果（上漲、下跌、不明確），不要加解釋：

新聞內容如下：
{joined_text}
"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        return f"分析失敗：{e}"

# ------------------------------------------------------------
# 📘 Firestore 分析流程
# ------------------------------------------------------------
def analyze_target(db, source_collection, company_name, result_collection, force_direction=False):
    """抓取新聞並送 Groq 分析後，將結果寫回 Firestore"""
    print(f"🚀 開始分析：{company_name}")

    docs = db.collection(source_collection).stream()
    news_list = []
    for doc in docs:
        data = doc.to_dict()
        title = data.get("title", "")
        content = data.get("content", "")
        full_text = f"{title}\n{content}".strip()
        if full_text:
            news_list.append(full_text)

    print(f"📄 共收集 {len(news_list)} 篇新聞")

    if not news_list:
        print(f"⚠️ {company_name} 沒有新聞可分析")
        return

    # 呼叫 Groq
    result = analyze_with_groq(news_list, company_name)
    print(f"🔍 分析結果：{result}")

    # 若 force_direction=True，避免出現「不明確」
    if force_direction and "不明確" in result:
        result = result.replace("不明確", "中性或略為上漲")

    # 寫入 Firestore
    today_str = datetime.now().strftime("%Y%m%d")
    result_ref = db.collection(result_collection).document(today_str)
    result_ref.set({
        "date": today_str,
        "result": result,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    print(f"✅ 已儲存至 Firestore：{result_collection}/{today_str}")

# ------------------------------------------------------------
# 🧭 主程式
# ------------------------------------------------------------
def main():
    db = get_db()

    # 台積電分析（一般模式）
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("\n" + "="*70 + "\n")

    # 鴻海分析（強制方向）
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon", force_direction=True)
    print("\n" + "="*70 + "\n")

    # 聯電分析（同鴻海邏輯 → 強制方向）
    analyze_target(db, "NEWS_UMC", "聯電", "Groq_result_UMC", force_direction=True)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
