import os
import regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from google.cloud import firestore
from dotenv import load_dotenv

# Groq client
try:
    from groq import Groq
except Exception:
    Groq = None

# ---------------- config ----------------
load_dotenv()
TAIWAN_TZ = timezone(timedelta(hours=8))

DAYS_BACK = int(os.getenv('DAYS_BACK', '7'))
NEWS_COLLECTION_TSMC = os.getenv('NEWS_COLLECTION_TSMC', 'NEWS')
NEWS_COLLECTION_FOX = os.getenv('NEWS_COLLECTION_FOX', 'NEWS_Foxxcon')
NEWS_COLLECTION_UMC = os.getenv('NEWS_COLLECTION_UMC', 'NEWS_UMC')
RESULT_COLLECTION = os.getenv('RESULT_COLLECTION', 'Groq_result')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if GROQ_API_KEY and Groq is not None:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# ---------------- dataclasses ----------------
@dataclass
class NewsItem:
    docid: str
    key: str
    title: str
    content: str
    price_change: str
    dt: datetime

# ---------------- helpers ----------------
def get_db():
    return firestore.Client()

def first_n_sentences(text: str, n: int = 3) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[。\.！!\?？；;])\s*', text.strip())
    return "".join(parts[:n]) + ("..." if len(parts) > n else "")

def parse_docid_time(doc_id: str):
    m = re.match(r"^(?P<ymd>\d{8})(?:_(?P<hms>\d{6}))?$", doc_id or "")
    if not m:
        return None
    ymd, hms = m.group('ymd'), m.group('hms') or '000000'
    try:
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=TAIWAN_TZ)
    except:
        return None

def collect_company_corpus(db, collection, days_back=DAYS_BACK) -> List[NewsItem]:
    today = datetime.now(TAIWAN_TZ).date()
    corpus = []
    for d in db.collection(collection).stream():
        dt = parse_docid_time(d.id)
        if not dt:
            continue
        if (today - dt.date()).days > days_back:
            continue
        data = d.to_dict() or {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            title = v.get('title', '')
            content = v.get('content', '')
            price_change = v.get('price_change', '')
            corpus.append(NewsItem(d.id, k, title, content, price_change, dt))
    return corpus

# ---------------- prompt assembly ----------------
def make_augmented_prompt(target: str, today_items: List[NewsItem], past_items: List[NewsItem]) -> str:
    header = f"你是一位專業的台股量化分析師，請根據下列資料並嚴格推論明日{target}股價方向，並給出一句 30 字內原因與情緒分數。\n\n"
    
    today_section = "【今日新聞（摘要 + 漲跌）】\n"
    for item in today_items:
        today_section += f"- {first_n_sentences(item.title,2)} {first_n_sentences(item.content,2)} 漲跌：{item.price_change}\n"

    past_section = "\n【過去新聞參考】\n"
    for item in past_items:
        past_section += f"- ({item.docid}) {first_n_sentences(item.title,2)} {first_n_sentences(item.content,2)} 漲跌：{item.price_change}\n"

    instr = ("\n請基於上面資料做判斷。輸出格式：\n"
             "明天{target}股價走勢：{上漲/微漲/微跌/下跌/不明確}（附符號）\n"
             "原因：一句 30 字內\n"
             "情緒分數：整數 -10~+10\n")
    return header + today_section + past_section + instr

# ---------------- Groq call ----------------
def call_groq_for_analysis(prompt: str):
    if groq_client is None:
        return f"明天股價走勢：不明確 ⚖️\n原因：Groq 未配置\n情緒分數：0"
    try:
        resp = groq_client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[
                {"role": "system", "content": "你是台股量化分析員，需根據新聞及漲跌資訊產生明確結論。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"明天股價走勢：不明確 ⚖️\n原因：Groq 呼叫失敗 {e}\n情緒分數：0"

# ---------------- main ----------------
def analyze_company_with_rag(db, collection, target, result_collection):
    corpus = collect_company_corpus(db, collection)
    if not corpus:
        print(f"{target}：無新聞可分析")
        return

    today = datetime.now(TAIWAN_TZ).date()
    today_items = [item for item in corpus if item.dt.date() == today]
    past_items = [item for item in corpus if item.dt.date() != today]

    prompt = make_augmented_prompt(target, today_items, past_items)
    analysis = call_groq_for_analysis(prompt)

    # write Firestore
    result_doc = {
        'timestamp': datetime.now(TAIWAN_TZ).isoformat(),
        'result': analysis,
        'today_count': len(today_items),
        'past_count': len(past_items)
    }
    try:
        db.collection(result_collection).document(datetime.now(TAIWAN_TZ).strftime('%Y%m%d')).set(result_doc)
    except Exception as e:
        print('[warning] Firestore 寫回失敗：', e)

    # write local file
    os.makedirs('results', exist_ok=True)
    fname = f"results/result_{datetime.now(TAIWAN_TZ).strftime('%Y%m%d')}.txt"
    with open(fname, 'a', encoding='utf-8') as f:
        f.write(f"======= {target} =======\n")
        f.write(analysis + '\n\n')
    print(f"[{target}] done. 今日 {len(today_items)} 則新聞，已存結果。")

def main():
    db = get_db()
    analyze_company_with_rag(db, NEWS_COLLECTION_TSMC, '台積電', RESULT_COLLECTION)
    analyze_company_with_rag(db, NEWS_COLLECTION_FOX, '鴻海', RESULT_COLLECTION + '_Foxxcon')
    analyze_company_with_rag(db, NEWS_COLLECTION_UMC, '聯電', RESULT_COLLECTION + '_UMC')

if __name__ == '__main__':
    main()
