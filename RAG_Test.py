# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
== 保留原判斷邏輯，不改準確率 ==
唯一變更：輸出格式改成單行精簡字串
"""
import os, signal, regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from google.cloud import firestore
from dotenv import load_dotenv
# 留著 Groq client 以防日後需要，但本版本不呼叫 LLM
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

# 硬規則加權（單次加成清單）
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
# 延後建立 Groq client（若需要再呼叫 get_groq_client）
def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)

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

# ---------- 解析 price_change ----------
def parse_price_change(raw: str) -> float:
    """
    解析格式範例：
    "+7.50 (+3.28%)" -> 0.0328
    "-1.20 (-0.42%)" -> -0.0042
    若無法解析則回傳 0.0
    """
    if not raw:
        return 0.0
    s = str(raw).replace(",", "").strip()
    m = re.search(r"\(([-+]?[\d\.]+)%\)", s) or re.search(r"([-+]?[\d\.]+)%", s)
    if not m:
        return 0.0
    try:
        return float(m.group(1)) / 100.0
    except:
        return 0.0

# ---------- Token ----------
def load_tokens(db):
    pos, neg = [], []
    try:
        for d in db.collection(TOKENS_COLLECTION).stream():
            data = d.to_dict() or {}
            pol = data.get("polarity", "").lower()
            ttype = data.get("type", "substr").lower()
            patt = data.get("pattern", "")
            note = data.get("note", "")
            try:
                w = float(data.get("weight", 1.0))
            except:
                w = 1.0
            if pol == "positive":
                pos.append(Token(pol, ttype, patt, w, note))
            elif pol == "negative":
                neg.append(Token(pol, ttype, patt, -abs(w), note))
    except Exception as e:
        # 若 tokens collection 不存在或讀取失敗，回傳空列表（程式仍可運行）
        print(f"[warning] load_tokens 失敗：{e}")
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
    if target not in aliases:
        return MatchResult(0.0, [])
    company_pattern = r"\b(?:" + "|".join(re.escape(a) for a in aliases.get(target, [])) + r")\b"
    if not re.search(company_pattern, norm):
        return MatchResult(0.0, [])

    for ttype, cre, w, note, patt in pos_c + neg_c:
        key = (patt, note)
        if key in seen:
            continue
        matched = cre.search(norm) if ttype == "regex" else (patt in norm)
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

# ---------- 市場調整與背離偵測 ----------
def adjust_by_market(avg_score: float, today_change: float) -> float:
    """
    根據今日漲跌幅調整平均分數。
    保守預設：
      - 當日大漲 (>= 3%)：+0.5
      - 當日中度漲跌 (|1%~3%|)：+/-0.2
      - 當日大跌 (<= -3%)：-0.5
    """
    if today_change >= 0.03:
        return avg_score + 0.5
    if today_change <= -0.03:
        return avg_score - 0.5
    if today_change >= 0.01:
        return avg_score + 0.2
    if today_change <= -0.01:
        return avg_score - 0.2
    return avg_score

def detect_divergence(avg_score: float, today_change: float) -> str:
    """
    背離檢查：
      - avg_score > 1 且 today_change < -2% -> 利多不漲
      - avg_score < -1 且 today_change > +2% -> 利空不跌
    """
    if avg_score > 1.0 and today_change < -0.02:
        return "利多不漲（疑似主力出貨）"
    if avg_score < -1.0 and today_change > 0.02:
        return "利空不跌（可能有隱性利多）"
    return "無明顯背離"

# ---------- 硬規則決策函式（替代 LLM） ----------
def decide_by_hard_rules(news_list: List[Tuple[str, float]], today_change: float, full_texts: List[str] = None, adjusted_avg: float = None, divergence: str = None) -> Tuple[str,int,List[str]]:
    """
    返回：
      - concise_str: 單行輸出（使用者要求格式）
      - mood_score_int: 情緒分數整數（-10..+10）
      - top_phrases: 用於原因欄的關鍵短語清單
    保留原始決策邏輯但改造輸出格式。
    """
    n = len(news_list)
    if n == 0:
        concise = "明天股價走勢：不明確 ⚖️ 原因：近三日無相關新聞。 情緒分數：0"
        return concise, 0, ["無新聞資料"]

    contributions = []
    reason_lines = []
    for idx, (title, weighted_score) in enumerate(news_list):
        base = 1.0 if weighted_score > 0 else (-1.0 if weighted_score < 0 else 0.0)
        add = 0.0
        txt = (full_texts[idx] if full_texts and idx < len(full_texts) else title).lower()

        # 單次正權重檢查
        for kw, v in HARD_WEIGHTS_POS.items():
            if kw in txt:
                add += v
                reason_lines.append(f"包含正向關鍵詞「{kw}」")
                break
        # 單次負權重檢查（優先負面）
        for kw, v in HARD_WEIGHTS_NEG.items():
            if kw in txt:
                add += v
                reason_lines.append(f"包含負向關鍵詞「{kw}」")
                break

        contrib = base + add
        contributions.append(contrib)
        # 簡短摘要一句
        sent = first_n_sentences(title, 1)
        reason_lines.append(f"新聞[{idx+1}]摘要：{sent}")

    total_score = sum(contributions)
    standardized = total_score / (n + 1)  # 舊標準化公式

    # 若有市場調整，採用 adjusted_avg（保持原程式設計）
    if adjusted_avg is not None:
        standardized = adjusted_avg
        reason_lines.append(f"已套用市場漲跌幅調整")

    # impact 分類（決定方向）
    if standardized >= 2.5:
        trend = "上漲"
        symbol = "🔼"
    elif standardized >= 1.0:
        trend = "微漲"
        symbol = "↗️"
    elif standardized > -1.0:
        trend = "微跌"
        symbol = "↘️"
    else:
        trend = "下跌"
        symbol = "🔽"

    # 今日走勢
    pct = round(today_change * 100, 2)
    trend_today = "上漲" if today_change > 0 else "下跌" if today_change < 0 else "平盤"

    # market_effect 判斷
    dir_sign = 1 if standardized > 0 else (-1 if standardized < 0 else 0)
    today_sign = 1 if today_change > 0 else (-1 if today_change < 0 else 0)
    if dir_sign != 0 and today_sign != 0:
        if dir_sign == today_sign:
            market_effect = "今日走勢與新聞方向同向。"
        else:
            market_effect = "今日走勢與新聞方向相反。"
    else:
        market_effect = "今日走勢或新聞方向中性。"

    # 情緒分數映射（-10~+10）
    mood_score = max(-10, min(10, int(round(standardized * 3))))

    # 形成 concise reason：從 reason_lines 中挑最重要的 3 條關鍵描述（去重）
    short_reasons = []
    seen = set()
    for line in reason_lines:
        # 提取有意義短語（去掉「新聞[...]摘要：」字樣）
        phrase = re.sub(r"新聞\[\d+\]摘要：", "", line).strip()
        # 取第一句話或前 60 字
        phrase = phrase.split("。")[0][:120]
        if phrase and phrase not in seen:
            short_reasons.append(phrase)
            seen.add(phrase)
        if len(short_reasons) >= 3:
            break

    # 若 short_reasons 空，放 fallback
    if not short_reasons:
        short_reasons = ["市場消息綜合影響"]

    # 合成單行輸出（符合使用者格式）
    reason_text = "；".join(short_reasons)
    concise_str = f"明天股價走勢：{trend} {symbol} 原因：{reason_text}。 情緒分數：{mood_score:+d}"

    # 若有 divergence，也把簡短說明加入 top_phrases，但不讓輸出變太長
    top_phrases = short_reasons.copy()
    if divergence and divergence != "無明顯背離":
        top_phrases.append(divergence)

    return concise_str, mood_score, top_phrases

# ---------- Groq analyze（只是包裝硬規則） ----------
def groq_analyze(news_list, target, avg_score, today_change, adjusted_avg=None, divergence=None):
    full_texts = [t for t, _ in news_list]
    concise, mood, top_phrases = decide_by_hard_rules(news_list, today_change, full_texts, adjusted_avg=adjusted_avg, divergence=divergence)
    # 在結果前加上 target 名稱
    # 結果已是單行，例如 "明天股價走勢：下跌 🔽 原因：... 情緒分數：-3"
    return concise.replace("明天股價走勢", f"明天{target}股價走勢", 1)

# ---------- 主分析（與原程式一致） ----------
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
            for k, v in data.items():
                if isinstance(v, dict) and "price_change" in v:
                    today_price_change = parse_price_change(v.get("price_change"))
                    break
            if today_price_change != 0.0:
                break
    except Exception:
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
        mood_score = 0
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
        if not SILENT_MODE:
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

        # 計算 avg_score（未調整）
        avg_score = sum(s for _, s in news_with_scores) / len(news_with_scores) if news_with_scores else 0.0

        # === 市場調整 & 背離偵測 ===
        adjusted_avg = adjust_by_market(avg_score, today_price_change)
        divergence = detect_divergence(avg_score, today_price_change)

        # 使用硬規則決策
        summary = groq_analyze(news_with_scores, target, avg_score, today_price_change, adjusted_avg=adjusted_avg, divergence=divergence)

        # 同步 mood_score（從 decide_by_hard_rules 取得更準確數值）
        # 重新呼叫以獲得 mood_score 與 top_phrases
        concise, mood_score, top_phrases = decide_by_hard_rules(news_with_scores, today_price_change, full_texts, adjusted_avg=adjusted_avg, divergence=divergence)

        # 本地存檔（保留較多細節於檔案，但 Firestore 只存 concise 字串）
        fname = f"result_{today.strftime('%Y%m%d')}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"======= {target} =======\n")
            f.write(f"今日漲跌 (程式讀取)：{round(today_price_change*100,2)}%\n")
            f.write(f"avg_score (原始)：{avg_score:+.4f}\n")
            f.write(f"avg_score (調整後)：{adjusted_avg:+.4f}\n")
            f.write(f"背離檢測：{divergence}\n\n")
            for docid, key, title, res, weight, full in top_news:
                hits_text = "\n".join([f"  {'+' if w>0 else '-'} {p}（{n}）" for p, w, n in res.hits])
                f.write(f"[{docid}#{key}]（{weight:.2f}x）\n標題：{first_n_sentences(title)}\n命中：\n{hits_text}\n\n")
            f.write(summary + "\n\n")

    # 印出與寫回 Firestore（只寫入 single-line concise string）
    print(summary + "\n")

    # Firestore 寫回（寫單行字串到 result collection under date doc）
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
        print("🚀 開始分析台股焦點股（準確率保留，輸出格式精簡）...\n")

    db = get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("=" * 70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
