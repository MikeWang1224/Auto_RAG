# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
準確率提升版（情緒融合 + 多層權重 + 語意補償）
✅ Firestore 寫回 + 本地 result.txt
✅ Groq 同時考慮每則情緒分數 + 平均分數
✅ 命中多則新聞時提升穩定度
✅ 新增：支援 3 天內新聞（延遲效應）
✅ 新增：從新聞抓今日漲跌 price_change，附在 Groq 分析原因
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
SCORE_THRESHOLD = 1.5

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

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

# ---------- 工具 ----------
DOCID_RE = re.compile(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$")

def get_db():
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

# ---------- Token ----------
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

# ---------- Scoring ----------
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

# ---------- Groq（情緒融合 + 準確率強化） ----------
def groq_analyze(news_list: List[Tuple[str, float]], target: str, avg_score: float, price_change: str = "") -> str:
    if not news_list:
        reason_text = f"近三日無相關新聞。今日漲跌：{price_change}" if price_change else "近三日無相關新聞"
        return f"明天{target}股價走勢：不明確 ⚖️\n原因：{reason_text}"

    # 將新聞內容與分數整合
    combined = "\n".join(f"{i+1}. ({s:+.2f}) {t}" for i, (t, s) in enumerate(news_list))

    prompt_text = f"""
你是一位金融新聞分析員。
請閱讀以下關於「{target}」最近三天的新聞摘要，
以「情緒融合模式」進行情緒總結與走勢預測：

1. 綜合新聞中的利多與利空情緒，給出整體情緒分數（-10 ~ +10）。
2. 若利多與利空勢均力敵，請回答「不明確 ⚖️」。
3. 若利多情緒明顯佔優（> +2），請回答「上漲 🔼」。
4. 若利空情緒明顯佔優（< -2），請回答「下跌 🔽」。
5. 附上簡短原因（40 字內），說明主導情緒的主要因素。
6. 可參考今日股價漲跌：{price_change}

整體平均情緒分數為 {avg_score:+.2f}。

以下是新聞摘要（含情緒分數）：
{combined}

請輸出格式如下：
明天{target}股價走勢：{{上漲／下跌／不明確}}（附符號）
原因：{{一句總結理由}}
情緒分數：{{整數（-10~10）}}
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是專業台股分析師，需綜合情緒與市場反應做出判斷。"},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.2,
            max_tokens=200,
            timeout=25,
        )
        ans = resp.choices[0].message.content.strip()
        ans = re.sub(r"\s+", " ", ans)

        # 提取走勢
        m_trend = re.search(r"(上漲|下跌|不明確|微漲|微跌)", ans)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲": "🔼", "微漲": "↗️", "微跌": "↘️", "下跌": "🔽", "不明確": "⚖️"}

        # 提取理由
        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+?)(?:情緒分數|$)", ans)
        reason = m_reason.group(1).strip() if m_reason else f"市場觀望。今日漲跌：{price_change}"

        # 提取情緒分數
        m_score = re.search(r"情緒分數[:：]?\s*(-?\d+)", ans)
        mood_score = int(m_score.group(1)) if m_score else 0

        return f"明天{target}股價走勢：{trend} {symbol_map.get(trend,'')}\n原因：{reason}\n情緒分數：{mood_score:+d}"

    except Exception as e:
        return f"明天{target}股價走勢：持平 ⚖️\n原因：Groq分析失敗({e})\n情緒分數：0"


# ---------- 主分析 ----------
def analyze_target(db, collection: str, target: str, result_field: str):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)

    today = datetime.now(TAIWAN_TZ).date()
    filtered, weighted_scores = [], []

    price_change = ""  # 只抓一次
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        news_date = dt.date()
        delta_days = (today - news_date).days

        if delta_days > 2:
            continue

        day_weight = {0:1.0, 1:0.85, 2:0.7}.get(delta_days,0.7)
        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            if not price_change:
                price_change = v.get("price_change", "")  # 只取第一筆

            title, content = v.get("title", ""), v.get("content", "")
            full = title + " " + content
            res = score_text(full, pos_c, neg_c, target)
            if not res.hits:
                continue

            token_weight = 1.0 + min(len(res.hits) * 0.05, 0.3)
            total_weight = day_weight * token_weight

            filtered.append((d.id, k, title, res, total_weight))
            weighted_scores.append(res.score * total_weight)

    if not filtered:
        print(f"{target}：近三日無新聞，交由 Groq 判斷。\n")
        summary = groq_analyze([], target, 0, price_change)
    else:
        filtered.sort(key=lambda x: abs(x[3].score * x[4]), reverse=True)
        top_news = filtered[:10]

        print(f"\n📰 {target} 近期重點新聞：")
        for docid, key, title, res, weight in top_news:
            print(f"[{docid}#{key}] ({weight:.2f}x, 分數={res.score:+.2f}) {title}")
            for p, w, n in res.hits:
                print(f"   {'+' if w>0 else '-'} {p}（{n}）")

        news_with_scores = [(t, res.score * weight) for _, _, t, res, weight in top_news]
        avg_score = sum(s for _, s in news_with_scores) / len(news_with_scores)
        summary = groq_analyze(news_with_scores, target, avg_score, price_change)

        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            for docid, key, title, res, weight in top_news:
                hits_text = "\n".join([f"  {'+' if w>0 else '-'} {p}（{n}）" for p, w, n in res.hits])
                f.write(f"[{docid}#{key}]（{weight:.2f}x）\n標題：{first_n_sentences(title)}\n命中：\n{hits_text}\n\n")
            f.write(summary + "\n\n")

    print(summary + "\n")

    try:
        db.collection(result_field).document(today.strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "result": summary,
        })
    except Exception as e:
        print(f"[warning] Firestore 寫回失敗：{e}")

# ---------- 主程式 ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股（準確率提升版）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
