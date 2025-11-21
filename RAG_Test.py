# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
準確率極致版（短期預測特化） - Context-aware + 去重新聞版
✅ 嚴格依據情緒分數決策
✅ 敏感詞加權（法說 / 財報 / 新品 / 停工等）
✅ 支援 3 日延遲效應
✅ Firestore 寫回 + 本地 result.txt
✅ 新增句型判斷，避免「重申／預期內」誤判
✅ 相同新聞內容去重
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

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

SENSITIVE_WORDS = {
    "法說": 1.5,
    "財報": 1.4,
    "新品": 1.3,
    "合作": 1.3,
    "併購": 1.4,
    "投資": 1.3,
    "停工": 1.6,
    "下修": 1.5,
    "利空": 1.5,
    "爆料": 1.4,
    "營收": 1.3,
    "展望": 1.2,
}

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
    m = re.match(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$", doc_id or "")
    if not m:
        return None
    ymd, hms = m.group("ymd"), m.group("hms") or "000000"
    try:
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

# ---------- 新增：解析 price_change ----------
def parse_price_change(raw: str) -> float:
    """
    解析格式範例：
    "+7.50 (+3.28%)" -> 0.0328
    "-1.20 (-0.42%)" -> -0.0042
    若無法解析則回傳 0.0
    """
    if not raw:
        return 0.0
    m = re.search(r"\(([-+]?[\d\.]+)%\)", raw)
    if not m:
        return 0.0
    try:
        return float(m.group(1)) / 100.0
    except:
        return 0.0

# ---------- Token ----------
def load_tokens(db):
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
    aliases = {"台積電": ["台積電", "tsmc", "2330"],
               "鴻海": ["鴻海", "foxconn", "2317", "富士康"],
               "聯電": ["聯電", "umc", "2303"]}
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

# ---------- Context-aware 調整 ----------
def adjust_score_for_context(text: str, base_score: float) -> float:
    if not text or base_score == 0:
        return base_score

    norm = text.lower()
    neutral_phrases = ["重申", "符合預期", "預期內", "中性看待", "無重大影響", "持平", "未變"]
    if any(p in norm for p in neutral_phrases):
        base_score *= 0.4

    positive_boost = ["創新高", "倍增", "大幅成長", "獲利暴增", "報喜"]
    negative_boost = ["暴跌", "下滑", "虧損", "停工", "下修", "裁員", "警訊"]
    if any(p in norm for p in positive_boost):
        base_score *= 1.3
    if any(p in norm for p in negative_boost):
        base_score *= 1.3

    return base_score

# ---------- 修改：Groq 判斷（加入 today_change） ----------
def groq_analyze(news_list, target, avg_score, today_change):
    """
    news_list: [(title, score), ...]
    avg_score: 平均情緒分數
    today_change: 今日實際漲跌幅 (float) 例如 +0.0328
    """
    if not news_list:
        return f"明天{target}股價走勢：不明確 ⚖️\n原因：近三日無相關新聞\n情緒分數：0"

    # 格式化新聞清單
    news_details = []
    for i, (title, score) in enumerate(news_list, 1):
        impact_desc = "正面" if score > 0 else "負面"
        news_details.append(f"{i}. 「{title}」 → {impact_desc}影響 ({score:+.2f})")
    combined = "\n".join(news_details)

    # 程式端建構原因
    pos_news = sorted([(t, s) for t, s in news_list if s > 0], key=lambda x: x[1], reverse=True)
    neg_news = sorted([(t, s) for t, s in news_list if s < 0], key=lambda x: x[1])
    top_pos = pos_news[:2]
    top_neg = neg_news[:2]

    sensitive_hits = []
    for t, s in news_list:
        tl = t.lower()
        for kw in SENSITIVE_WORDS.keys():
            if kw in tl:
                sensitive_hits.append((t, kw))
                break

    reason_lines = []
    if top_pos:
        rp = "; ".join([f"「{t}」({s:+.2f})" for t, s in top_pos])
        reason_lines.append(f"主要利多：{rp}")
    if top_neg:
        rn = "; ".join([f"「{t}」({s:+.2f})" for t, s in top_neg])
        reason_lines.append(f"主要利空：{rn}")
    if sensitive_hits:
        sh = "; ".join([f"「{t}」(含 {kw})" for t, kw in sensitive_hits])
        reason_lines.append(f"敏感議題強化影響：{sh}")

    reason_lines.append(f"綜合來看平均情緒分數為 {avg_score:+.2f}，反映正負新聞交錯，但仍偏向{'多頭' if avg_score>0 else '空頭' if avg_score<0 else '中性'}。")

    # 今日漲跌字串
    pct = round(today_change * 100, 2)
    trend_today = "上漲" if today_change > 0 else "下跌" if today_change < 0 else "平盤"
    reason_lines.append(f"今日市場真實走勢：{trend_today}（{pct}%），作為市場即時反應指標。")

    constructed_reason = "；".join(reason_lines)

    # 構造 prompt，明確要求模型比較新聞與今日市場反應
    prompt = f"""
你是一位專業台股金融分析師，請依據以下「{target}」近三日新聞摘要與今日市場走勢，
嚴格推論明日股價方向，並給出詳細原因。請務必在「原因」段落中：
1) 逐條評估每則新聞對股價的正/負貢獻（可採上方列出的格式），
2) 指出主要利多與主要利空（各至多兩項），
3) 若新聞含敏感詞（法說、財報、新品、停工等），請說明其放大效果，
4) 評估今日市場走勢（已提供）是否「強化」或「抵銷」新聞發出的訊號，
5) 最後給出一句整體總結（40字以內）。

下面是程式端的預先整理（請在說明中引用或修正）：
---- 程式端摘要開始 ----
{combined}

程式端快速判斷（供你參考，非最終結論）：
{constructed_reason}
---- 程式端摘要結束 ----

【今日市場即時走勢（程式提供）】
- 今日股價：{trend_today}（{pct}%）

請根據上面內容並結合你的金融常識產出以下格式（所有欄位都要出現）：
明天{target}股價走勢：{{上漲／微漲／微跌／下跌／不明確}}（附符號）
原因：{{詳盡說明，包含每則新聞貢獻、主要利多/利空、敏感詞影響、今日走勢如何影響明日判斷與簡短總結}}
情緒分數：{{整數 -10~+10}}

注意：如果你採用程式端提供的「主要利多/利空」或「敏感議題」，請在原因中明確標示你是否同意，並說明理由。
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "你是台股量化分析員，需根據新聞情緒與當日盤勢生成明確趨勢和詳細原因。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.12,
            max_tokens=450,
        )
        ans = resp.choices[0].message.content.strip()
        ans = re.sub(r"\s+", " ", ans)

        # 解析 model 回傳（保留 trend / model 原因 / model 分數）
        m_trend = re.search(r"(上漲|微漲|微跌|下跌|不明確)", ans)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲": "🔼", "微漲": "↗️", "微跌": "↘️", "下跌": "🔽", "不明確": "⚖️"}

        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+?)(?:情緒分數|$)", ans)
        model_reason = m_reason.group(1).strip() if m_reason and m_reason.group(1).strip() else None

        m_score = re.search(r"情緒分數[:：]?\s*(-?\d+)", ans)
        mood_score = int(m_score.group(1)) if m_score else max(-10, min(10, int(round(avg_score * 3))))

        # 組合最終 reason
        if model_reason:
            short_model = len(model_reason) < 30 or model_reason.lower().strip() in ["整體平均", "綜合各新聞正負影響形成市場短線觀望。"]
            if short_model:
                final_reason = constructed_reason
            else:
                final_reason = model_reason + "；" + constructed_reason
        else:
            final_reason = constructed_reason

        if len(final_reason) > 600:
            final_reason = final_reason[:590].rsplit("。", 1)[0] + "。 (摘要...)"

        return f"明天{target}股價走勢：{trend} {symbol_map.get(trend,'')}\n原因：{final_reason}\n情緒分數：{mood_score:+d}"

    except Exception as e:
        # fallback
        fallback_reason = constructed_reason + "（Groq 呼叫失敗，使用程式端預先生成之分析。）"
        return f"明天{target}股價走勢：不明確 ⚖️\n原因：{fallback_reason}\n情緒分數：{max(-10, min(10, int(round(avg_score * 3)))):+d}"

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()

    filtered, weighted_scores = [], []
    today_price_change = 0.0

    # ---------- 先掃一次 collection 取得今日 price_change（若有） ----------
    try:
        for d in db.collection(collection).stream():
            dt = parse_docid_time(d.id)
            if not dt:
                continue
            if dt.date() != today:
                continue
            data = d.to_dict() or {}
            # data 可能包含多個 key，每個 key 是一篇新聞的 dict
            for k, v in data.items():
                if isinstance(v, dict) and "price_change" in v:
                    today_price_change = parse_price_change(v.get("price_change"))
                    break
            if today_price_change != 0.0:
                break
    except Exception:
        # 若讀取過程有問題，保留 today_price_change = 0.0
        today_price_change = 0.0

    # ---------- 原有新聞打分流程（保留） ----------
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        delta_days = (today - dt.date()).days
        if delta_days > 2:
            continue

        day_weight = 1.0 if delta_days == 0 else 0.85 if delta_days == 1 else 0.7
        data = d.to_dict() or {}

        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            title, content = v.get("title", ""), v.get("content", "")
            full = title + " " + content
            res = score_text(full, pos_c, neg_c, target)
            if not res.hits:
                continue

            adj_score = adjust_score_for_context(full, res.score)
            token_weight = 1.0 + min(len(res.hits) * 0.05, 0.3)
            impact = 1.0 + sum(w * 0.05 for k_sens, w in SENSITIVE_WORDS.items() if k_sens in full)
            total_weight = day_weight * token_weight * impact

            filtered.append((d.id, k, title, full, res, total_weight))
            weighted_scores.append(adj_score * total_weight)

    # ---------- 無新聞 fallback ----------
    if not filtered:
        summary = groq_analyze([], target, 0, today_price_change)
    else:
        # 去重新聞
        seen_text = set()
        top_news = []
        for docid, key, title, full, res, weight in sorted(filtered, key=lambda x: abs(x[4].score * x[5]), reverse=True):
            news_text = normalize(full)
            if news_text in seen_text:
                continue
            seen_text.add(news_text)
            top_news.append((docid, key, title, res, weight))
            if len(top_news) >= 10:
                break

        # 輸出新聞摘要（console）
        print(f"\n📰 {target} 近期重點新聞（含衝擊）:")
        for docid, key, title, res, weight in top_news:
            impact_val = sum(w for k_sens, w in SENSITIVE_WORDS.items() if k_sens in title)
            print(f"[{docid}#{key}] ({weight:.2f}x, 分數={res.score:+.2f}, 衝擊={1+impact_val/10:.2f}) {title}")
            for p, w, n in res.hits:
                sign = "+" if w>0 else "-"
                print(f"   {sign} {p}（{n}）")

        news_with_scores = [(t, res.score * weight) for _, _, t, res, weight in top_news]
        avg_score = sum(s for _, s in news_with_scores) / len(news_with_scores)
        summary = groq_analyze(news_with_scores, target, avg_score, today_price_change)

        # 本地存檔
        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            for docid, key, title, res, weight in top_news:
                hits_text = "\n".join([f"  {'+' if w>0 else '-'} {p}（{n}）" for p, w, n in res.hits])
                f.write(f"[{docid}#{key}]（{weight:.2f}x）\n標題：{first_n_sentences(title)}\n命中：\n{hits_text}\n\n")
            f.write(summary + "\n\n")

    print(summary + "\n")

    # Firestore 寫回
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
        print("🚀 開始分析台股焦點股（準確率極致版）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
