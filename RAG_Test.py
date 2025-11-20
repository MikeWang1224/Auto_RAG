import os
import json
import regex as re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from google.cloud import firestore
from dotenv import load_dotenv

# optional providers
try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss
except Exception:
    faiss = None

# Groq client
try:
    from groq import Groq
except Exception:
    Groq = None

# OpenAI (optional embed provider)
try:
    import openai
except Exception:
    openai = None

# ---------------- config ----------------
load_dotenv()
TAIWAN_TZ = timezone(timedelta(hours=8))
EMB_PROVIDER = os.getenv('EMB_PROVIDER', 'groq')  # 'groq' or 'openai'
OPENAI_EMBED_MODEL = os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-small')
GROQ_EMBED_MODEL = os.getenv('GROQ_EMBED_MODEL', 'groq-embedding-1')  # 修改為可用模型
TOP_K = int(os.getenv('TOP_K', '5'))
DAYS_BACK = int(os.getenv('DAYS_BACK', '7'))

TOKENS_COLLECTION = os.getenv('TOKENS_COLLECTION', 'bull_tokens')
NEWS_COLLECTION_TSMC = os.getenv('NEWS_COLLECTION_TSMC', 'NEWS')
NEWS_COLLECTION_FOX = os.getenv('NEWS_COLLECTION_FOX', 'NEWS_Foxxcon')
NEWS_COLLECTION_UMC = os.getenv('NEWS_COLLECTION_UMC', 'NEWS_UMC')
RESULT_COLLECTION = os.getenv('RESULT_COLLECTION', 'Groq_result')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if GROQ_API_KEY and Groq is not None:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

if OPENAI_API_KEY and openai is not None:
    openai.api_key = OPENAI_API_KEY

# ---------------- dataclasses ----------------
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

# ---------------- helpers ----------------
def get_db():
    return firestore.Client()

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

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

# ---------------- token loader ----------------
def load_tokens(db):
    pos, neg = [], []
    col = db.collection(TOKENS_COLLECTION)
    for d in col.stream():
        data = d.to_dict() or {}
        pol = data.get('polarity', '').lower()
        ttype = data.get('type', 'substr').lower()
        patt = data.get('pattern', '')
        note = data.get('note', '')
        w = float(data.get('weight', 1.0))
        if pol == 'positive':
            pos.append(Token(pol, ttype, patt, w, note))
        elif pol == 'negative':
            neg.append(Token(pol, ttype, patt, -abs(w), note))
    return pos, neg

def compile_tokens(tokens: List[Token]):
    compiled = []
    for t in tokens:
        if t.ttype == 'regex':
            try:
                compiled.append(('regex', re.compile(t.pattern, re.I), t.weight, t.note, t.pattern))
            except:
                continue
        else:
            compiled.append(('substr', None, t.weight, t.note, t.pattern))
    return compiled

# ---------------- scoring ----------------
def score_text(text: str, pos_c, neg_c, target: str = None) -> MatchResult:
    norm = normalize(text).lower()
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
        matched = cre.search(norm) if ttype == 'regex' else (patt.lower() in norm)
        if matched:
            score += w
            hits.append((patt, w, note))
            seen.add(key)
    return MatchResult(score, hits)

def adjust_score_for_context(text: str, base_score: float) -> float:
    if not text or base_score == 0:
        return base_score
    norm = text.lower()
    neutral_phrases = ["重申", "符合預期", "預期內", "中性看待", "無重大影響", "持平", "未變"]
    if any(p in norm for p in neutral_phrases):
        base_score *= 0.45
    positive_boost = ["創新高", "倍增", "大幅成長", "獲利暴增", "報喜"]
    negative_boost = ["暴跌", "下滑", "虧損", "停工", "下修", "裁員", "警訊"]
    if any(p in norm for p in positive_boost):
        base_score *= 1.25
    if any(p in norm for p in negative_boost):
        base_score *= 1.25
    return base_score

# ---------------- embeddings ----------------
def embed_texts_groq(texts: List[str]):
    if groq_client is None:
        raise RuntimeError('Groq client not configured')
    resp = groq_client.embeddings.create(model=GROQ_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_texts_openai(texts: List[str]):
    if openai is None:
        raise RuntimeError('openai package not available')
    resp = openai.Embedding.create(input=texts, model=OPENAI_EMBED_MODEL)
    return [d['embedding'] for d in resp['data']]

def embed_texts(texts: List[str]):
    if EMB_PROVIDER == 'openai' and OPENAI_API_KEY:
        return embed_texts_openai(texts)
    else:
        return embed_texts_groq(texts)

# ---------------- vector utils ----------------
def cosine_sim(a, b):
    if np is None:
        raise RuntimeError('numpy required for cosine_sim')
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ---------------- firestore helpers ----------------
def collect_company_corpus(db, collection, days_back=DAYS_BACK):
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
            corpus.append((d.id, k, title, content, dt))
    return corpus

def ensure_embedding_on_firestore_item(db, collection, docid, key, text):
    doc_ref = db.collection(collection).document(docid)
    try:
        doc = doc_ref.get().to_dict() or {}
        item = doc.get(key, {})
        if 'embedding' in item and item.get('embedding'):
            return item.get('embedding')
    except Exception:
        pass
    emb = embed_texts([text])[0]
    try:
        doc_ref.update({f"{key}.embedding": emb})
    except Exception:
        pass
    return emb

# ---------------- build index ----------------
def build_index(embs):
    if faiss is not None and np is not None:
        xb = np.array(embs).astype('float32')
        dim = xb.shape[1]
        index = faiss.IndexFlatIP(dim)
        norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8
        xb_norm = xb / norms
        index.add(xb_norm)
        return index
    else:
        return None

# ---------------- retrieval ----------------
def retrieve_most_similar(query_text: str, corpus, corpus_embs, k=TOP_K):
    if not corpus:
        return []
    q_emb = embed_texts([query_text])[0]
    if faiss is not None and np is not None and corpus_embs is not None:
        qv = np.array(q_emb).astype('float32')
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        D, I = build_index(corpus_embs).search(qv.reshape(1, -1), k)
        idxs = [int(i) for i in I[0] if i != -1]
        return idxs
    else:
        scores = []
        for i, emb in enumerate(corpus_embs):
            sim = cosine_sim(q_emb, emb)
            scores.append((sim, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]

# ---------------- RAG prompt assembly ----------------
def make_augmented_prompt(target: str, today_items: List[Tuple], retrieved: List[Tuple]):
    header = f"你是一位專業的台股量化分析師，請根據下列資料並嚴格推論明日{target}股價方向，並給出一句 30 字內原因與情緒分數。\n\n"
    today_section = "【今日新聞（摘要）】\n"
    for docid, key, title, content, dt in today_items:
        today_section += f"- {first_n_sentences(title,2)} {first_n_sentences(content,2)}\n"

    past_section = "\n【過去相似新聞（供參考）】\n"
    for docid, key, title, content, dt in retrieved:
        past_section += f"- ({docid}) {first_n_sentences(title,2)} {first_n_sentences(content,2)}\n"

    instr = ("\n請基於上面資料做判斷。輸出格式：\n明天{target}股價走勢：{上漲/微漲/微跌/下跌/不明確}（附符號）\n原因：一句 30 字內\n情緒分數：整數 -10~+10\n")
    return header + today_section + past_section + instr

# ---------------- Groq call ----------------
def call_groq_for_analysis(prompt: str):
    if groq_client is None:
        return f"明天股價走勢：不明確 ⚖️\n原因：Groq 未配置\n情緒分數：0"
    try:
        resp = groq_client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[
                {"role": "system", "content": "你是台股量化分析員，需根據情緒分數規則產生明確結論。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"明天股價走勢：不明確 ⚖️\n原因：Groq 呼叫失敗 {e}\n情緒分數：0"

# ---------------- main RAG flow per company ----------------
def analyze_company_with_rag(db, collection, target, result_collection):
    pos, neg = load_tokens(db)
    pos_c, neg_c = compile_tokens(pos), compile_tokens(neg)

    corpus = collect_company_corpus(db, collection, days_back=DAYS_BACK)
    if not corpus:
        print(f"{target}：近 {DAYS_BACK} 日無新聞")
        return

    corpus_texts = [t[2] + "\n" + t[3] for t in corpus]
    corpus_embs = []
    for (docid, key, title, content, dt), text in zip(corpus, corpus_texts):
        emb = None
        try:
            doc = db.collection(collection).document(docid).get().to_dict() or {}
            item = doc.get(key, {})
            emb = item.get('embedding')
        except Exception:
            emb = None
        if not emb:
            emb = ensure_embedding_on_firestore_item(db, collection, docid, key, title + '\n' + content)
        corpus_embs.append(emb)

    today = datetime.now(TAIWAN_TZ).date()
    today_items = [item for item in corpus if item[4].date() == today]

    query_text = f"{target} 今日新聞 會如何影響明日股價？摘要："
    for docid, key, title, content, dt in today_items:
        query_text += first_n_sentences(title, 2) + " ; "

    retrieved_idxs = retrieve_most_similar(query_text, corpus, corpus_embs, k=TOP_K)
    retrieved = [corpus[i] for i in retrieved_idxs if corpus[i] not in today_items]

    prompt = make_augmented_prompt(target, today_items, retrieved)
    analysis = call_groq_for_analysis(prompt)

    result_doc = {
        'timestamp': datetime.now(TAIWAN_TZ).isoformat(),
        'result': analysis,
        'query': query_text,
        'retrieved_count': len(retrieved)
    }
    try:
        db.collection(result_collection).document(datetime.now(TAIWAN_TZ).strftime('%Y%m%d')).set(result_doc)
    except Exception as e:
        print('[warning] Firestore 寫回失敗：', e)

    os.makedirs("results", exist_ok=True)
    fname = f"results/result_{datetime.now(TAIWAN_TZ).strftime('%Y%m%d')}.txt"
    with open(fname, 'a', encoding='utf-8') as f:
        f.write(f"======= {target} =======\n")
        f.write(analysis + '\n\n')
    print(f"[{target}] done. Retrieved {len(retrieved)} items. Result saved.")

# ---------------- entrypoint ----------------
def main():
    db = get_db()
    analyze_company_with_rag(db, NEWS_COLLECTION_TSMC, '台積電', RESULT_COLLECTION)
    analyze_company_with_rag(db, NEWS_COLLECTION_FOX, '鴻海', RESULT_COLLECTION + '_Foxxcon')
    analyze_company_with_rag(db, NEWS_COLLECTION_UMC, '聯電', RESULT_COLLECTION + '_UMC')

if __name__ == '__main__':
    main()
