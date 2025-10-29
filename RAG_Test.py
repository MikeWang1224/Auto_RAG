# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG 版：台積電 + 鴻海 + 聯電）
改良版：加強錯誤處理、環境檢查、日誌與輸出穩定性
"""
import os
import signal
import regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv

# load env once
load_dotenv()

# ---------- 設定 ----------
SILENT_MODE = True            # 設為 False 可看到詳細日誌
MAX_DISPLAY_NEWS = 5
TOKENS_COLLECTION = "bull_tokens"
NEWS_COLLECTION_TSMC = "NEWS_TSMC"
NEWS_COLLECTION_FOX = "NEWS_FOXCONN"
NEWS_COLLECTION_UMC = "NEWS_UMC"
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "2"))
TAIWAN_TZ = timezone(timedelta(hours=8))
RESULT_DIR = "result"

def log(msg: str):
    if not SILENT_MODE:
        print(msg)

# Ctrl+C 安全停止
STOP = False
def _sigint_handler(signum, frame):
    global STOP
    STOP = True
    print("\n[info] 偵測到 Ctrl+C，將安全停止…")
signal.signal(signal.SIGINT, _sigint_handler)

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

def normalize(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def shorten_text(t: str, n=200):
    return t[:n] + "…" if len(t) > n else t

def first_n_sentences(text: str, n: int = 3) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[。\.！!\?？；;])\s*', text.strip())
    parts = [p for p in parts if p.strip()]
    if not parts:
        return text.strip()
    joined = "".join(parts[:n])
    if not re.search(r'[。\.！!\?？；;]\s*$', joined):
        joined += "..."
    return joined

def parse_docid_time(doc_id: str) -> Optional[datetime]:
    """支援 20251018 或 20251018_064229 兩種格式；回傳帶時區的 datetime 或 None"""
    doc_id = (doc_id or "").strip()
    m = DOCID_RE.match(doc_id)
    if not m:
        return None
    ymd = m.group("ymd")
    hms = m.group("hms") or "000000"
    try:
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except Exception:
        return None

# ---------- Firestore 相關 ----------
def get_db():
    """延後 import google.cloud.firestore，並捕捉未設定憑證情況"""
    try:
        from google.cloud import firestore
    except Exception as e:
        raise RuntimeError("google-cloud-firestore 未安裝或無法匯入：請確認環境並安裝 google-cloud-firestore") from e
    try:
        return firestore.Client()
    except Exception as e:
        raise RuntimeError("建立 Firestore client 失敗：請確認 GOOGLE_APPLICATION_CREDENTIALS 或 GCP 環境設定") from e

def load_tokens(db) -> Tuple[List[Token], List[Token]]:
    pos, neg = [], []
    try:
        for d in db.collection(TOKENS_COLLECTION).stream():
            data = d.to_dict() or {}
            pol = (data.get("type") or "").lower()
            ttype = (data.get("method") or "substr").lower()
            patt = str(data.get("pattern") or "")
            note = str(data.get("note") or "")
            try:
                w = float(data.get("weight", 1.0))
            except:
                w = 1.0
            if not patt or pol not in ("positive", "negative"):
                continue
            (pos if pol == "positive" else neg).append(Token(pol, ttype, patt, w, note))
    except Exception as e:
        log(f"[warn] 讀取 tokens 失敗：{e}")
    return pos, neg

def load_news_items(db, col_name: str, days: int) -> List[Dict]:
    """從 collection 撈 news documents（文件內每個 field 可能是不同來源），只取最近 days 天"""
    items, seen = [], set()
    now = datetime.now(TAIWAN_TZ)
    start = now - timedelta(days=days)
    try:
        for d in db.collection(col_name).stream():
            dt = parse_docid_time(d.id)
            # 若 doc id 無法解析，保留（視為近期），但避免過久
            if dt and dt < start:
                continue
            data = d.to_dict() or {}
            for k, v in data.items():
                if not isinstance(v, dict):
                    continue
                title, content = str(v.get("title") or ""), str(v.get("content") or "")
                if not title and not content:
                    continue
                uniq = f"{title}|{content}"
                if uniq in seen:
                    continue
                seen.add(uniq)
                items.append({"id": f"{d.id}#{k}", "title": title, "content": content, "ts": dt})
    except Exception as e:
        log(f"[warn] load_news_items 讀取 {col_name} 失敗：{e}")
    items.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=TAIWAN_TZ), reverse=True)
    return items

# ---------- Token 編譯與打分 ----------
def compile_tokens(tokens: List[Token]):
    """回傳 list of (kind, compiled_or_none, weight, note, raw_pattern)"""
    out = []
    for t in tokens:
        w = t.weight if t.polarity == "positive" else -abs(t.weight)
        if t.ttype == "regex":
            try:
                cre = re.compile(t.pattern, flags=re.IGNORECASE)
                out.append(("regex", cre, w, t.note, t.pattern))
            except Exception:
                log(f"[warn] 無法編譯 regex pattern: {t.pattern}")
                continue
        else:
            out.append(("substr", t.pattern.lower(), w, t.note, t.pattern))
    return out

def score_text(text: str, pos_c, neg_c, target: str = None) -> MatchResult:
    """針對單一新聞 text 判定分數（會先檢查是否與 target 有關）"""
    norm = normalize(text)
    score, hits, seen_keys = 0.0, [], set()

    aliases = {
        "台積電": ["台積電", "tsmc", "2330"],
        "鴻海": ["鴻海", "hon hai", "2317", "foxconn", "富士康"],
        "聯電": ["聯電", "umc", "2303"],
    }
    # all_aliases 為扁平小寫清單（去重）
    all_aliases = list({a.lower() for arr in aliases.values() for a in arr} | {"台積電","鴻海","聯電"})
    target_aliases = [a.lower() for a in (aliases.get(target, []) + [target] if target else [])]

    # 若沒有任何 target 關鍵字出現在全文（快速過濾）
    if target and not any(tk in norm for tk in target_aliases):
        return MatchResult(0.0, [])

    # 以句子為單位進行檢查
    sentences = re.split(r'(?<=[。\.！!\?？；;])\s*', norm)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # 若句中完全沒有任何公司 alias，跳過
        if not any(alias in sent for alias in all_aliases):
            continue
        # 對段落（句子）直接用 token 做比對，避免 segment 切得太細造成遺漏
        for ttype, patt, w, note, raw in pos_c + neg_c:
            key = (raw, note)
            if key in seen_keys:
                continue
            matched = False
            if ttype == "regex":
                try:
                    if patt.search(sent):
                        matched = True
                except Exception:
                    continue
            else:
                if patt in sent:
                    matched = True
            if matched:
                score += w
                hits.append((raw, w, note))
                seen_keys.add(key)
    return MatchResult(score, hits)

# ---------- Groq 呼叫（可注入 client） ----------
def make_groq_client():
    try:
        from groq import Groq
    except Exception as e:
        raise RuntimeError("groq 套件無法匯入：請安裝 groq SDK") from e
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("環境變數 GROQ_API_KEY 未設定，無法呼叫 Groq")
    return Groq(api_key=api_key)

def prepare_news_for_llm(news_items: List[Dict]) -> str:
    parts = []
    for i, it in enumerate(news_items, 1):
        title = first_n_sentences(it.get("title",""), 2)
        content = shorten_text(it.get("content",""), 500)
        parts.append(f"新聞 {i}：\n標題：{title}\n內容：{content}\n")
    return "\n".join(parts)

def groq_analyze(client, texts: List[Dict], target: str, token_summary: str = "") -> str:
    combined = prepare_news_for_llm(texts)
    prompt = f"""你是一位台灣股市研究員。根據以下新聞與打分摘要，判斷「明天{target}股價」最可能走勢。
請只回覆以下兩行格式（不要多餘文字）：

明天{target}股價走勢：<上漲 / 下跌 / 不明確>
原因：<40字以內，一句話簡潔說明主要理由>

打分摘要（來自 Firestore bull_tokens）：
{token_summary}

新聞摘要：
{combined}
"""
    try:
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL","llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": "你是專業股市新聞分析員，回答簡潔準確。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=220,
        )
        raw = resp.choices[0].message.content.strip()
        cleaned = re.sub(r"^```(?:\w+)?|```$", "", raw).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        m_trend = re.search(r"(上漲|下跌|不明確)", cleaned)
        trend = m_trend.group(1) if m_trend else "不明確"
        symbol_map = {"上漲": "🔼", "下跌": "🔽", "不明確": "⚠️"}
        trend_with_symbol = f"{trend} {symbol_map.get(trend, '')}"

        m_reason = re.search(r"(?:原因|理由)[:：]?\s*(.+)", cleaned)
        reason_text = m_reason.group(1) if m_reason else cleaned
        sentences = re.split(r"[。.!！；;]", reason_text)
        short_reason = "，".join(sentences[:2]).strip()
        short_reason = re.sub(r"\s+", " ", short_reason)[:40].strip("，,。")
        return f"明天{target}股價走勢：{trend_with_symbol}\n原因：{short_reason}"
    except Exception as e:
        return f"[error] Groq 呼叫失敗：{e}"

# ---------- 分析 ----------
def analyze_target(db, news_col: str, target: str, result_col: str):
    try:
        pos, neg = load_tokens(db)
        pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
        items = load_news_items(db, news_col, LOOKBACK_DAYS)
    except Exception as e:
        log(f"[error] 讀取資料失敗：{e}")
        return

    # 排除明確不相關關鍵字
    exclude_keywords = ["intel", "輝達", "nvidia", "日月光"]
    def is_excluded(it):
        txt = (it.get("title","") + " " + it.get("content","")).lower()
        return any(k.lower() in txt for k in exclude_keywords)
    items = [it for it in items if not is_excluded(it)]

    if not items:
        log(f"[info] {target} 在最近 {LOOKBACK_DAYS} 天無新聞或皆被過濾。")
        return

    filtered, terminal_logs = [], []
    for it in items:
        if STOP:
            break
        text_for_score = (it.get("content") or it.get("title") or "")
        res = score_text(text_for_score, pos_c, neg_c, target)
        if abs(res.score) >= SCORE_THRESHOLD and res.hits:
            filtered.append((it, res))
            trend = "✅ 明日可能大漲" if res.score > 0 else "❌ 明日可能下跌"
            hits_text_lines = [f"  {'+' if w>0 else '-'} {patt}（{note}）" for patt, w, note in res.hits]
            truncated_title = first_n_sentences(it.get("title",""), 3)
            terminal_logs.append(f"[{it['id']}]\n標題：{truncated_title}\n{trend}\n命中：\n" + "\n".join(hits_text_lines) + "\n")

    for t in terminal_logs[:MAX_DISPLAY_NEWS]:
        print(t)

    token_summary = "\n".join([
        f"新聞：{first_n_sentences(x[0].get('title',''),1)} 分數：{x[1].score:+.2f} 命中：{', '.join([n for _,_,n in x[1].hits])}"
        for x in filtered
    ])

    # 準備呼叫 groq
    try:
        client = make_groq_client()
    except Exception as e:
        log(f"[warn] 無法建立 Groq client：{e}")
        summary = f"[error] 無法建立 Groq client：{e}"
    else:
        summary = groq_analyze(client, [x[0] for x in filtered], target, token_summary)

    print(summary)

    # 儲存本機檔案
    os.makedirs(RESULT_DIR, exist_ok=True)
    fname_safe = re.sub(r"[^\w\-]", "_", target)
    local_path = os.path.join(RESULT_DIR, f"{fname_safe}_{datetime.now(TAIWAN_TZ).strftime('%Y%m%d_%H%M%S')}.txt")
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write("\n".join(terminal_logs))
            f.write("\n" + "="*60 + "\n")
            f.write(summary + "\n")
        log(f"[info] 結果已寫入 {local_path}")
    except Exception as e:
        log(f"[warn] 無法寫入本地結果：{e}")

    # 嘗試上傳到 Firestore（以日期為 doc id）
    try:
        docid = datetime.now(TAIWAN_TZ).strftime("%Y%m%d")
        db.collection(result_col).document(docid).set({
            "timestamp": datetime.now(TAIWAN_TZ),
            "result": summary,
            "items_count": len(filtered),
        })
        log(f"[info] 已上傳結果至 Firestore: {result_col}/{docid}")
    except Exception as e:
        log(f"[warn] 上傳 Firestore 失敗：{e}")

# ---------- 主程式 ----------
def main():
    try:
        db = get_db()
    except Exception as e:
        print(f"[error] 初始化 Firestore 失敗：{e}")
        return

    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("\n" + "="*70 + "\n")
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("\n" + "="*70 + "\n")
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__ == "__main__":
    main()
