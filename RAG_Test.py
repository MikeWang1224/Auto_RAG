# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
準確率極致版（短期預測特化） - Context-aware + 去重新聞版
✅ 嚴格依據情緒分數決策（硬規則量化版）
✅ 敏感詞加權（單次加成）
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
# 留著 Groq client 以防日後需要，但本版本不呼叫模型
from groq import Groq

# ---------- 設定 ----------
SILENT_MODE = True
TAIWAN_TZ = timezone(timedelta(hours=8))

TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS"
NEWS_COLLECTION_FOX = "NEWS_Foxxcon"
NEWS_COLLECTION_UMC = "NEWS_UMC"

# 內部敏感詞表（舊版保留，主加權規則另設）
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

# 硬規則加權（單次加成清單，照你要求的 mapping）
HARD_WEIGHTS_POS = {
    "財報": 1.5,
    "法說": 1.5,
    "展望": 1.5,
    "資本支出": 1.5,
    "訂單": 1.2,
    "擴產": 1.2,
    "爆單": 1.2,
    "漲價": 1.2,
}
HARD_WEIGHTS_NEG = {
    "停工": -1.5,
    "裁員": -1.5,
    "虧損": -1.5,
    "下修": -1.5,
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
# 建立 client（目前不呼叫）
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

# ---------- 新：硬規則決策函式（替代 LLM） ----------
def decide_by_hard_rules(news_list: List[Tuple[str, float]], today_change: float, full_texts: List[str] = None) -> str:
    """
    news_list: [(title, score_weighted), ...]
    today_change: 當日漲跌（float）
    full_texts: 對應每則新聞的完整文字（可選，用來檢查是否包含指定關鍵詞）
    返回：格式化的分析字串（與原本 groq_analyze 相容）
    規則：
      - 每則新聞 base: 正面 +1.0 / 負面 -1.0 / 0 為中性
      - 若新聞含硬權重關鍵詞，單次加成（正向或負向）
      - 最終標準化分數 = sum(each_contribution) / (N + 1)
      - impact 分類閾值：
          >= 2.5 -> impact 1（強烈利多）
          1.0 <= score < 2.5 -> impact 2（偏多）
          -1.0 < score < 1.0 -> impact 3（盤整偏空）
          <= -1.0 -> impact 4（強烈利空）
    """
    n = len(news_list)
    if n == 0:
        return "明天股價走勢：不明確 ⚖️\n原因：近三日無相關新聞\n情緒分數：0"

    contributions = []
    reason_lines = []
    for idx, (title, weighted_score) in enumerate(news_list):
        # base polarity from weighted_score sign
        base = 1.0 if weighted_score > 0 else (-1.0 if weighted_score < 0 else 0.0)
        add = 0.0
        txt = (full_texts[idx] if full_texts and idx < len(full_texts) else title).lower()

        # 單次正權重檢查
        for kw, v in HARD_WEIGHTS_POS.items():
            if kw in txt:
                add += v
                reason_lines.append(f"新聞[{idx+1}] 包含正向關鍵詞「{kw}」，加成 {v:+.2f}")
                break  # 單次加成
        # 單次負權重檢查（優先負面）
        for kw, v in HARD_WEIGHTS_NEG.items():
            if kw in txt:
                add += v  # v 已經是負數
                reason_lines.append(f"新聞[{idx+1}] 包含負向關鍵詞「{kw}」，加成 {v:+.2f}")
                break

        contrib = base + add
        contributions.append(contrib)
        reason_lines.append(f"新聞[{idx+1}]：標題/內容摘要「{first_n_sentences(title,1)}」，基礎貢獻 {base:+.2f}，加權後 {contrib:+.2f}")

    total_score = sum(contributions)
    standardized = total_score / (n + 1)  # 按你指定的標準化公式

    # impact 分類
    if standardized >= 2.5:
        impact = 1
        trend = "上漲"
        symbol = "🔼"
    elif standardized >= 1.0:
        impact = 2
        trend = "微漲"
        symbol = "↗️"
    elif standardized > -1.0:
        impact = 3
        trend = "微跌"
        symbol = "↘️"
    else:
        impact = 4
        trend = "下跌"
        symbol = "🔽"

    # 今日走勢與新聞方向關聯
    pct = round(today_change * 100, 2)
    trend_today = "上漲" if today_change > 0 else "下跌" if today_change < 0 else "平盤"
    # 判斷是否強化或抵銷（簡單判斷：標準化分數方向與今日走勢方向）
    dir_sign = 1 if standardized > 0 else (-1 if standardized < 0 else 0)
    today_sign = 1 if today_change > 0 else (-1 if today_change < 0 else 0)
    if dir_sign != 0 and today_sign != 0:
        if dir_sign == today_sign:
            market_effect = "今日走勢與新聞方向同向，市場走勢強化新聞信號。"
        else:
            market_effect = "今日走勢與新聞方向相反，市場走勢可能已提前消化或抵銷新聞影響。"
    else:
        market_effect = "今日走勢或新聞方向中性，無明顯強化/抵銷判斷。"

    # 情緒分數映射（-10~+10），利用 standardized * 3（並 clamp）
    mood_score = max(-10, min(10, int(round(standardized * 3))))

    # 構造最終原因（限制長度但保留細項）
    detail_reason = "\n".join(reason_lines)
    summary_reason = f"標準化分數 {standardized:+.2f}；{market_effect} (今日漲跌 {trend_today} {pct}%)"

    final_text = (
        f"明天股價走勢：{trend} {symbol}\n"
        f"原因：{summary_reason}\n"
        f"細節：\n{detail_reason}\n"
        f"情緒分數：{mood_score:+d}"
    )
    return final_text

# ---------- 修改：Groq 判斷（已改為硬規則） ----------
def groq_analyze(news_list, target, avg_score, today_change):
    """
    新版本：使用硬規則（decide_by_hard_rules）替代 LLM。
    news_list: [(title, score), ...]（程式端已乘上權重）
    avg_score: 平均情緒分數（保留傳入以便未來使用）
    today_change: 今日實際漲跌幅 (float)
    """
    # news_list 內的 title 已提供，我們也嘗試把 title 當做 full_texts 送入判斷函式
    full_texts = [t for t, _ in news_list]
    result = decide_by_hard_rules(news_list, today_change, full_texts)
    # 在結果前加上 target 方便辨識
    return result.replace("明天股價走勢", f"明天{target}股價走勢", 1)

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
            top_news.append((docid, key, title, res, weight, full))
            if len(top_news) >= 10:
                break

        # 輸出新聞摘要（console）
        print(f"\n📰 {target} 近期重點新聞（含衝擊）:")
        for docid, key, title, res, weight, full in top_news:
            impact_val = sum(w for k_sens, w in SENSITIVE_WORDS.items() if k_sens in title)
            print(f"[{docid}#{key}] ({weight:.2f}x, 分數={res.score:+.2f}, 衝擊={1+impact_val/10:.2f}) {title}")
            for p, w, n in res.hits:
                sign = "+" if w>0 else "-"
                print(f"   {sign} {p}（{n}）")

        # 構造 news_with_scores 供硬規則使用（保留 title 及加權後分數）
        news_with_scores = []
        full_texts = []
        for _, _, title, res, weight, full in top_news:
            news_with_scores.append((title, res.score * weight))
            full_texts.append(full)

        # 計算 avg_score（保留）
        avg_score = sum(s for _, s in news_with_scores) / len(news_with_scores)

        # 使用硬規則決策替代原本的 LLM 呼叫
        # groq_analyze 內會呼叫 decide_by_hard_rules
        summary = groq_analyze(news_with_scores, target, avg_score, today_price_change)

        # 本地存檔
        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            for docid, key, title, res, weight, full in top_news:
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
        print("🚀 開始分析台股焦點股（準確率極致版 - 硬規則）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
