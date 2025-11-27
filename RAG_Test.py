# -*- coding: utf-8 -*-
"""
股票新聞分析工具（多公司 RAG + 背離版：台積電 + 鴻海 + 聯電）
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

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SENSITIVE_WORDS = {
    "法說": 1.5, "財報": 1.4, "新品": 1.3, "合作": 1.3, "併購": 1.4,
    "投資": 1.3, "停工": 1.6, "下修": 1.5, "利空": 1.5, "爆料": 1.4,
    "營收": 1.3, "展望": 1.2
}

HARD_WEIGHTS_POS = {
    "財報": 1.5, "法說": 1.5, "展望": 1.5, "資本支出": 1.5,
    "訂單": 1.2, "擴產": 1.2, "爆單": 1.2, "漲價": 1.2
}

HARD_WEIGHTS_NEG = {
    "停工": -1.5, "裁員": -1.5, "虧損": -1.5, "下修": -1.5
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
    if not m: return None
    ymd, hms = m.group("ymd"), m.group("hms") or "000000"
    try: return datetime.strptime(ymd+hms, "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except: return None

def parse_price_change(raw: str) -> float:
    if not raw: return 0.0
    m = re.search(r"\(([-+]?[\d\.]+)%\)", raw)
    if not m: return 0.0
    try: return float(m.group(1))/100.0
    except: return 0.0

# ---------- Token ----------
def load_tokens(db):
    pos, neg = [], []
    for d in db.collection(TOKENS_COLLECTION).stream():
        data = d.to_dict() or {}
        pol, ttype, patt, note = data.get("polarity","").lower(), data.get("type","substr").lower(), data.get("pattern",""), data.get("note","")
        w = float(data.get("weight",1.0))
        if pol=="positive": pos.append(Token(pol, ttype, patt, w, note))
        elif pol=="negative": neg.append(Token(pol, ttype, patt, -abs(w), note))
    return pos, neg

def compile_tokens(tokens: List[Token]):
    compiled = []
    for t in tokens:
        if t.ttype=="regex":
            try: compiled.append(("regex", re.compile(t.pattern, re.I), t.weight, t.note, t.pattern))
            except: continue
        else:
            compiled.append(("substr", None, t.weight, t.note, t.pattern.lower()))
    return compiled

# ---------- Scoring ----------
def score_text(text: str, pos_c, neg_c, target: str = None) -> MatchResult:
    norm, score, hits, seen = normalize(text), 0.0, [], set()
    aliases = {"台積電":["台積電","tsmc","2330"], "鴻海":["鴻海","foxconn","2317","富士康"], "聯電":["聯電","umc","2303"]}
    company_pattern = "|".join(re.escape(a) for a in aliases.get(target,[]))
    if not re.search(company_pattern, norm): return MatchResult(0.0,[])
    for ttype, cre, w, note, patt in pos_c + neg_c:
        key = (patt, note)
        if key in seen: continue
        matched = cre.search(norm) if ttype=="regex" else patt in norm
        if matched: score+=w; hits.append((patt,w,note)); seen.add(key)
    return MatchResult(score,hits)

def adjust_score_for_context(text: str, base_score: float) -> float:
    if not text or base_score==0: return base_score
    norm = text.lower()
    neutral=["重申","符合預期","預期內","中性看待","無重大影響","持平","未變"]
    pos_boost=["創新高","倍增","大幅成長","獲利暴增","報喜"]
    neg_boost=["暴跌","下滑","虧損","停工","下修","裁員","警訊"]
    if any(p in norm for p in neutral): base_score*=0.4
    if any(p in norm for p in pos_boost): base_score*=1.3
    if any(p in norm for p in neg_boost): base_score*=1.3
    return base_score

def adjust_by_market(avg_score: float, today_change: float) -> float:
    if today_change>=0.03: return avg_score+0.5
    if today_change<=-0.03: return avg_score-0.5
    if today_change>=0.01: return avg_score+0.2
    if today_change<=-0.01: return avg_score-0.2
    return avg_score

def detect_divergence(avg_score: float, today_change: float) -> str:
    if avg_score>1.0 and today_change<-0.02: return "利多不漲（疑似主力出貨）"
    if avg_score<-1.0 and today_change>0.02: return "利空不跌（可能有隱性利多）"
    return "無明顯背離"

# ---------- RAG 背離分析 ----------
def rag_divergence_analysis(news_list, divergence, today_change, target):
    if not news_list: return f"明天{target}股價走勢：不明確 ⚖️\n原因：近三日無新聞\n情緒分數：0"
    prompt = f"請依據以下新聞標題及摘要，分析 {target} 股價明日走勢，並考慮背離訊號 {divergence}，今日漲跌 {today_change*100:.2f}%:\n"
    for idx,(docid,title,full,res,weight,full_text) in enumerate(news_list):
        prompt += f"\n[{idx+1}] 標題: {title}\n摘要: {first_n_sentences(full_text,2)}\n"

    resp = client.chat.completions.create(
        messages=[
            {"role":"system","content":"你是一個股票分析助手，請生成中文分析報告"},
            {"role":"user","content":prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.7,
    )
    return resp.choices[0].message.content

# ---------- 主分析 ----------
def analyze_target(db, collection, target, result_field):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)
    today = datetime.now(TAIWAN_TZ).date()
    filtered=[]
    today_price_change=0.0

    # 取得今日漲跌
    try:
        for d in db.collection(collection).stream():
            dt=parse_docid_time(d.id)
            if not dt or dt.date()!=today: continue
            data=d.to_dict() or {}
            for k,v in data.items():
                if isinstance(v,dict) and "price_change" in v:
                    today_price_change=parse_price_change(v.get("price_change"))
                    break
            if today_price_change!=0.0: break
    except: today_price_change=0.0

    # 評分新聞
    for d in db.collection(collection).stream():
        dt=parse_docid_time(d.id)
        if not dt: continue
        delta_days=(today-dt.date()).days
        if delta_days>2: continue
        day_weight=1.0 if delta_days==0 else 0.85 if delta_days==1 else 0.7
        data=d.to_dict() or {}
        for k,v in data.items():
            if not isinstance(v,dict): continue
            title, content=v.get("title",""), v.get("content","")
            full=title+" "+content
            res=score_text(full,pos_c,neg_c,target)
            if not res.hits: continue
            adj_score=adjust_score_for_context(full,res.score)
            token_weight=1.0+min(len(res.hits)*0.05,0.3)
            impact=1.0+sum(w*0.05 for k_sens,w in SENSITIVE_WORDS.items() if k_sens in full)
            total_weight=day_weight*token_weight*impact
            filtered.append((d.id,k,title,res,total_weight,full))

    # 去重並選前10則新聞
    seen=set(); top_news=[]
    for docid,key,title,res,weight,full in sorted(filtered,key=lambda x:abs(x[3].score*x[4]),reverse=True):
        txt=normalize(full)
        if txt in seen: continue
        seen.add(txt); top_news.append((docid,key,title,res,weight,full))
        if len(top_news)>=10: break

    # 計算 avg_score、調整、背離
    if top_news:
        news_with_scores=[(title,res.score*weight) for _,_,title,res,weight,_ in top_news]
        avg_score=sum(s for _,s in news_with_scores)/len(news_with_scores)
        adjusted_avg=adjust_by_market(avg_score,today_price_change)
        divergence=detect_divergence(avg_score,today_price_change)
        summary=rag_divergence_analysis(top_news,divergence,today_price_change,target)
    else:
        summary=f"明天{target}股價走勢：不明確 ⚖️\n原因：近三日無新聞\n情緒分數：0"

    # 寫入 results
    fname=os.path.join(RESULTS_DIR,f"{target}_{today.strftime('%Y%m%d')}.txt")
    with open(fname,"w",encoding="utf-8") as f:
        f.write(summary)

    print(summary+"\n")

    # Firestore 寫回
    try:
        db.collection(result_field).document(today.strftime("%Y%m%d")).set({
            "timestamp": datetime.now(TAIWAN_TZ).isoformat(),
            "result": summary
        })
    except Exception as e:
        print(f"[warning] Firestore 寫回失敗：{e}")

# ---------- 主程式 ----------
def main():
    if not SILENT_MODE:
        print("🚀 開始分析台股焦點股（RAG + 背離版）...\n")
    db=get_db()
    analyze_target(db, NEWS_COLLECTION_TSMC, "台積電", "Groq_result")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_FOX, "鴻海", "Groq_result_Foxxcon")
    print("="*70)
    analyze_target(db, NEWS_COLLECTION_UMC, "聯電", "Groq_result_UMC")

if __name__=="__main__":
    main()
