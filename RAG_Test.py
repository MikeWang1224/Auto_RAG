# -*- coding: utf-8 -*-
"""
股票新聞分析工具（完整 RAG 版 + 全文打分 + Groq 分析 + Token 傾向顯示 + Firestore 回傳結果）
已修改：讓 Groq 輸出穩定為兩行簡潔格式（A 格式）
"""

import os
import signal
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime, timedelta, timezone
import regex as re
import numpy as np
import faiss
from google.cloud import firestore
from dotenv import load_dotenv
import time
from groq import Groq
from rich import print as rprint  # rich 專用

# ---------- 讀 .env ----------
def load_env():
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
        print(f"[info] 已載入 .env：{os.path.abspath('.env')}")
    else:
        load_dotenv(override=True)
        print("[info] 未找到 .env，改用系統環境變數")

load_env()

# ---------- 確認金鑰 ----------
GOOGLE_CRED = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GOOGLE_CRED and os.path.exists(GOOGLE_CRED):
    print(f"[info] 已找到 Firestore 認證金鑰：{GOOGLE_CRED}")
else:
    print("[warn] 未找到 Firestore 認證金鑰（GOOGLE_APPLICATION_CREDENTIALS），可能無法寫入 Firebase。")

# ---------- 參數 ----------
TOKENS_COLLECTION = os.getenv("FIREBASE_TOKENS_COLLECTION", "bull_tokens")
NEWS_COLLECTION   = os.getenv("FIREBASE_NEWS_COLLECTION", "NEWS")
SCORE_THRESHOLD   = float(os.getenv("SCORE_THRESHOLD", "3.0"))
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "3"))
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "mistral")
TITLE_LEN         = 20
GROQ_RESULT_COLLECTION = "Groq_result"

TAIWAN_TZ = timezone(timedelta(hours=8))
DOCID_RE  = re.compile(r"^(?P<ymd>\d{8})_(?P<hms>\d{6})$")

# ---------- 中斷控制 ----------
STOP = False
def _sigint_handler(signum, frame):
    global STOP
    if not STOP:
        STOP = True
        print("\n[info] 偵測到 Ctrl+C，將儘快安全停止…")
    else:
        raise KeyboardInterrupt
signal.signal(signal.SIGINT, _sigint_handler)

# ---------- 小工具 ----------
def normalize(text: str) -> str:
    if not text: return ""
    return re.sub(r"\s+", " ", text.strip()).lower()

def shorten_text(text: str, max_len: int = 500) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text

def parse_docid_time(doc_id: str):
    m = DOCID_RE.match(doc_id)
    if not m: return None
    try:
        return datetime.strptime(m.group("ymd")+m.group("hms"), "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

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

# ---------- Firestore ----------
def get_db() -> firestore.Client:
    db = firestore.Client()
    print(f"[info] GCP 專案：{db.project}")
    return db

def save_groq_result_to_firestore(db: firestore.Client, result_text: str):
    """將 Groq 分析結果回傳到 Firestore"""
    try:
        ts = datetime.now(TAIWAN_TZ).strftime("%Y%m%d_%H%M%S")
        doc_ref = db.collection(GROQ_RESULT_COLLECTION).document(ts)
        doc_ref.set({
            "timestamp": datetime.now(TAIWAN_TZ),
            "result": result_text,
        })
        print(f"[info] 已將分析結果寫入 Firestore：{GROQ_RESULT_COLLECTION}/{ts}")
    except Exception as e:
        print(f"[error] 寫入 Firestore 失敗：{e}")

def load_tokens(db: firestore.Client, col_name: str) -> Tuple[List[Token], List[Token]]:
    pos, neg = [], []
    for d in db.collection(col_name).stream():
        data = d.to_dict() or {}
        polarity = (data.get("polarity") or "").lower()
        ttype    = (data.get("type") or "substr").lower()
        patt     = str(data.get("pattern") or "")
        note     = str(data.get("note") or "")
        try:
            weight = float(data.get("weight", 1.0))
        except:
            weight = 1.0
        if not patt or polarity not in ("positive","negative"): continue
        (pos if polarity=="positive" else neg).append(Token(polarity, ttype, patt, weight, note))
    return pos, neg

def load_news_items(db: firestore.Client, col_name: str, lookback_days: int) -> List[Dict]:
    items: List[Dict] = []
    seen = set()
    now_tw = datetime.now(TAIWAN_TZ)
    start_tw = now_tw - timedelta(days=lookback_days)
    for d in db.collection(col_name).stream():
        dt = parse_docid_time(d.id)
        if dt and dt < start_tw:
            continue
        data = d.to_dict() or {}
        for key, val in data.items():
            if not (isinstance(key,str) and key.startswith("news_") and isinstance(val,dict)):
                continue
            title   = str(val.get("title") or "").strip()
            content = str(val.get("content") or "").strip()
            if not title and not content:
                continue
            uniq_key = f"{title}|{content}"
            if uniq_key in seen:
                continue
            seen.add(uniq_key)
            items.append({
                "id": f"{d.id}#{key}",
                "title": title,
                "content": content,
                "ts": dt
            })
    items.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=TAIWAN_TZ), reverse=True)
    return items

# ---------- Token 編譯 / 打分 ----------
def compile_tokens(tokens: List[Token]):
    out = []
    for t in tokens:
        w = t.weight
        if t.polarity == "negative": w = -abs(w)
        if t.ttype == "regex":
            try:
                cre = re.compile(t.pattern, flags=re.IGNORECASE)
                out.append(("regex", cre, w, t.note, t.pattern))
            except:
                continue
        else:
            out.append(("substr", None, w, t.note, t.pattern.lower()))
    return out

def score_text(text: str, pos_compiled, neg_compiled) -> MatchResult:
    norm = normalize(text)
    score = 0.0
    hits = []
    seen_patterns = set()
    for ttype, cre, w, note, patt in pos_compiled + neg_compiled:
        key = (ttype, patt.lower() if ttype=="substr" else patt)
        if key in seen_patterns:
            continue
        matched = False
        if ttype=="regex" and cre.search(norm):
            matched = True
        elif ttype=="substr" and patt in norm:
            matched = True
        if matched:
            score += w
            hits.append((patt, w, note))
            seen_patterns.add(key)
    return MatchResult(score, hits)

# ---------- Groq 分析 ----------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def prepare_news_for_llm(news_items: List[str]) -> str:
    out_lines = []
    for i, txt in enumerate(news_items, 1):
        out_lines.append(f"新聞 {i}：\n{shorten_text(txt, 200)}\n")
    return "\n".join(out_lines)

def ollama_analyze(texts: List[str], target: str = "台積電", retries=3, delay=1.5) -> str:
    """
    產生 A 格式（兩行）自然語言輸出：
    明天{target}股價走勢：<上漲 / 下跌 / 不明確>
    原因：<100字以內原因>

    此函式會：
    - 加強 prompt 要求模型回覆 A 格式
    - 清洗 model 回傳（去掉 code fence、多餘空行）
    - 用較寬鬆的 regex 嘗試解析；若仍失敗，使用第一句話或前100字作為原因
    最終保證回傳兩行格式。
    """
    global STOP
    combined_text = prepare_news_for_llm(texts)

    # Prompt 強調回覆格式（自然語言 A 格式）
    prompt = f"""你是一位專業的台灣股市研究員。請根據以下新聞判斷「明天{target}股價」最可能的走勢。
請**只回覆**以下兩行格式（不要多做說明、不要增加額外文字）：

明天{target}股價走勢：<上漲 / 下跌 / 不明確>
原因：<100字以內，簡要說明主要理由>

範例（有效）：
明天{target}股價走勢：上漲
原因：AI 訂單回溫帶動高階製程出貨，法人維持買超。

新聞摘要：
{combined_text}

⚠️ 注意：輸出**必須**包含「明天{target}股價走勢：」跟「原因：」兩行，trend 只能是 上漲、下跌 或 不明確。Reason 最多 100 字。
"""

    for i in range(retries):
        if STOP:
            return "[stopped] 使用者中斷"
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "你是一個專業的股市新聞分析助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # 穩定輸出
                max_tokens=300,
            )
            raw_result = response.choices[0].message.content.strip()

            # 清洗：去除 code fence（``` 或 ```json）及首尾空白
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_result).strip()

            # 把多個空行壓成一個換行，並去掉前後多餘空白
            cleaned = re.sub(r"\r\n", "\n", cleaned)
            cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()

            # 嘗試用嚴格的正則抓兩行：trend 與 reason
            # 接受「：」或 ":"，也允許前面有少量雜訊
            m = re.search(
                r"(明天\s*"+re.escape(target)+r"\s*股價走勢[:：]\s*(上漲|下跌|不明確))\s*[\r\n]+原因[:：]?\s*([^\n]{1,100})",
                cleaned
            )
            if m:
                trend = m.group(2).strip()
                reason = m.group(3).strip()
                reason = re.sub(r"\s+", " ", reason)[:100].strip()
                return f"明天{target}股價走勢：{trend}\n原因：{reason if reason else '未提供明確理由'}"

            # 寬鬆 fallback：找第一個出現的 上漲/下跌/不明確
            m_trend = re.search(r"\b(上漲|下跌|不明確)\b", cleaned)
            trend = m_trend.group(1) if m_trend else "不明確"

            # 找「原因：...」或「理由：...」
            m_reason = re.search(r"(?:原因|理由)[:：]?\s*([^\n]{1,200})", cleaned)
            if m_reason:
                reason = m_reason.group(1).strip()
                reason = re.sub(r"\s+", " ", reason)[:100].strip()
                return f"明天{target}股價走勢：{trend}\n原因：{reason if reason else '未提供明確理由'}"

            # 最後 fallback：取回 cleaned 的前 100 字（去掉標題行）
            # 若 cleaned 本身就是很多行，取第二行或第一句
            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            reason_candidate = ""
            if len(lines) >= 2:
                # 優先取第二行（假設第一行為走勢）
                reason_candidate = lines[1][:100]
            elif len(lines) == 1:
                # 嘗試從單行中截取一句話
                reason_candidate = lines[0][:100]
            else:
                reason_candidate = ""

            reason_candidate = re.sub(r"\s+", " ", reason_candidate).strip()
            if not reason_candidate:
                reason_candidate = "未取得原因或格式不符"

            return f"明天{target}股價走勢：{trend}\n原因：{reason_candidate}"

        except Exception as e:
            print(f"[warn] Groq 呼叫失敗 ({i+1}/{retries})：{e}")
            time.sleep(delay)

    return "[error] Groq 呼叫失敗"

# ---------- 主程式 ----------
def main():
    global STOP
    today = datetime.now(TAIWAN_TZ).strftime("%Y%m%d")
    output_file = f"result_{today}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        def out(msg):
            print(msg)
            f.write(msg + "\n")

        out(f"[info] 設定：tokens='{TOKENS_COLLECTION}', news='{NEWS_COLLECTION}', threshold={SCORE_THRESHOLD}")

        db = get_db()
        pos, neg = load_tokens(db, TOKENS_COLLECTION)
        pos_c = compile_tokens(pos)
        neg_c = compile_tokens(neg)

        items = load_news_items(db, NEWS_COLLECTION, LOOKBACK_DAYS)
        if not items:
            out("[info] NEWS 無資料可分析。")
            return

        filtered: List[Tuple[Dict, MatchResult]] = []
        for item in items:
            if STOP: break
            text_for_scoring = item["content"] or item["title"]
            res = score_text(text_for_scoring, pos_c, neg_c)
            if abs(res.score) >= SCORE_THRESHOLD:
                filtered.append((item, res))

        out(f"[info] 過濾後新聞: {len(filtered)} / {len(items)}")
        if not filtered or STOP:
            out("[info] 無新聞達到閾值或已中斷，結束")
            return

        for item, res in filtered:
            short_title = item["title"] if item["title"] else item["content"]
            out(f"\n[{item['id']}] {short_title}")
            if res.score >= SCORE_THRESHOLD:
                out("✅ 明日可能大漲")
            elif res.score <= -SCORE_THRESHOLD:
                out("⚠️ 明日可能大跌")
            else:
                out("—")
            if res.hits:
                out("命中：")
                for patt, w, note in res.hits:
                    why = f"（{note}）" if note else ""
                    out(f"  {'+' if w>0 else '-'} {patt} {why}")

        summary = ollama_analyze([n[0]["content"] or n[0]["title"] for n in filtered], target="台積電")
        out("\n--- Groq 生成分析 ---")
        out(summary)

        # 上傳結果至 Firestore
        save_groq_result_to_firestore(db, summary)

if __name__ == "__main__":
    main()
