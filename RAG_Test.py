# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
準確率提升版（情緒融合 + 多層權重 + 語意補償）
✅ Firestore 寫回 + 本地 result.txt
✅ Groq 同時考慮每則情緒分數 + 平均分數
✅ 命中多則新聞時提升穩定度
✅ 新增：支援 3 天內新聞（延遲效應）
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
def groq_analyze(news_list: List[Tuple[str, float]], target: str, avg_score: float) -> str:
    if not news_list:
        return f"明天{target}股價走勢：不明確 ⚖️\n原因：近三日無相關新聞"

    combined = "\n".join(f"{i+1}. ({s:+.2f}) {t}" for i, (t, s) in enumerate(news_list))
    prompt = f"""你是一位台股分析師。
根據以下{target}的新聞摘要與情緒分數：
- 平均分數大於 +0.5 代表偏多（可能上漲）
- 平均分數小於 -0.5 代表偏空（可能下跌）
- 若在 -0.5 至 +0.5 之間則市場觀望。

整體平均情緒分數為 {avg_score:+.2f}。
請綜合所有新聞內容與分數，預測明天{target}的股價方向，避免回答「不明確」除非真的極為中性。

格式如下：
明天{target}股價走勢：<上漲 / 微漲 / 微跌 / 下跌 / 不明確>
原因：<一句話40字內>

{combined}
"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是專業股市分析師，語氣簡潔明確。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=120,
            timeout=25,
        )
        ans = resp.choices[0].message.content.strip()
        ans = re.sub(r"\s+", " ", ans)
        m_trend = re.search(r"(上漲|微漲|微跌|下跌|不明確)", ans)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲": "🔼", "微漲": "↗️", "微跌": "↘️", "下跌": "🔽", "不明確": "⚖️"}
        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+)", ans)
        reason = m_reason.group(1) if m_reason else "市場觀望"
        return f"明天{target}股價走勢：{trend} {symbol_map.get(trend,'')}\n原因：{reason[:40]}"
    except Exception as e:
        return f"明天{target}股價走勢：持平 ⚖️\n原因：Groq分析失敗({e})"

# ---------- 主分析 ----------
def analyze_target(db, collection: str, target: str, result_field: str):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)

    today = datetime.now(TAIWAN_TZ).date()
    filtered, weighted_scores = [], []

    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        news_date = dt.date()
        delta_days = (today - news_date).days

        # 延長時間窗（支援 1~2 天延遲效應，最多取 3 天內）
        if delta_days > 2:
            continue

        # 根據時間給不同權重（越久影響越弱）
        if delta_days == 0:
            day_weight = 1.0   # 今日新聞權重最高
        elif delta_days == 1:
            day_weight = 0.85  # 昨日稍弱
        else:
            day_weight = 0.7   # 前天再弱一些

        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
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
        summary = groq_analyze([], target, 0)
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
        summary = groq_analyze(news_with_scores, target, avg_score)

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
