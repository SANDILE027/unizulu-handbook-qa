#!/usr/bin/env python3
import os, json, re, numpy as np, streamlit as st, torch, faiss
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer

BASE_DIR   = "/home/sandile/NLPP"
MODEL_DIR  = os.path.join(BASE_DIR, "models", "model_C", "final_model")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CODE_RE = re.compile(r"\b([0-9][A-Z]{2,4}[0-9]{2,3}(?:/[A-Z0-9]+)?)\b", re.IGNORECASE)

INTENT_ORDER = ["code","title","credits","nqf","aim","outcomes","admission","duration","aps","faculty","intake","prereq"]
INTENT_MAP = {
    "code":      ["code","ikhodi","module code","ikhodi yemojuli"],
    "title":     ["title","isihloko","name","igama","called"],
    "credits":   ["credit","credits","ikhredithi","amakhredithi","ama khredithi"],
    "nqf":       ["nqf","level","izinga","ibanga","nqf level"],
    "aim":       ["aim","inhloso","purpose","inhlos","inhlso","yini inhloso","ithini inhloso"],
    "outcomes":  ["outcome","outcomes","imiphumela","imphumela","okulindelekile"],
    "admission": ["admission","izimfanelo zokungena","requirement","requirements","entry"],
    "duration":  ["duration","iminyaka","years","how long","how many years"],
    "aps":       ["aps","minimum aps","amaphuzu e-aps","points required"],
    "faculty":   ["faculty","ifakhalthi","department","umnyango","which faculty"],
    "intake":    ["intake","ukuqaliswa","when does","start","semester"],
    "prereq":    ["prerequisite","prerequisites","izimfanelo zangaphambili","required modules"],
}

def detect_intent(query):
    q = query.lower()
    for field in INTENT_ORDER:
        if any(kw in q for kw in INTENT_MAP[field]):
            return field
    return "general"

@st.cache_resource
def load_ner_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_data
def load_data():
    modules    = json.load(open(os.path.join(BASE_DIR,"normalized_modules.json"), encoding="utf-8"))
    programmes = json.load(open(os.path.join(BASE_DIR,"normalized_programmes.json"), encoding="utf-8"))
    return modules, programmes

@st.cache_data
def build_indexes(modules, programmes):
    code_index = {}
    for doc in modules + programmes:
        code = (doc.get("code") or "").upper().strip()
        if code:
            code_index[code] = doc
            code_index[code.split("/")[0]] = doc
    chunks, metas = [], []
    for doc in modules:
        text = f"{doc.get('code','')} {doc.get('title','')} {doc.get('department','')} {str(doc.get('aim',''))[:150]}"
        chunks.append(text.strip()); metas.append({"type":"module","doc":doc})
    for doc in programmes:
        text = f"{doc.get('name','')} {doc.get('faculty','')} {doc.get('degree','')}"
        chunks.append(text.strip()); metas.append({"type":"programme","doc":doc})
    return code_index, chunks, metas

@st.cache_data
def build_faiss(_embedder, chunks):
    embs = _embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    return index

def clean(v):
    return str(v).strip() if v and str(v).strip() not in ("","NOT FOUND","None","none","not specified") else None

def build_answer(doc, intent, is_prog):
    code    = doc.get("code","")
    title   = doc.get("title","") or doc.get("name","")
    credits = doc.get("credits","")
    nqf     = doc.get("nqf_level","")
    aim     = doc.get("aim","") or doc.get("description","")
    content = doc.get("content","")
    dept    = doc.get("department","")
    prereq  = doc.get("prerequisites","")
    faculty = doc.get("faculty","")
    duration= doc.get("duration_years","")
    aps     = doc.get("aps","")
    intake  = doc.get("intake","")
    admission = doc.get("admission_requirements",[])
    degree  = doc.get("degree","")
    label   = f"**{code} — {title}**" if code else f"**{title}**"

    if intent == "code":      return f"{label}\n\n🔑 **Code:** {code or 'not recorded'}"
    if intent == "title":     return f"**{code}** is titled: **{title}**"
    if intent == "credits":   return f"{label}\n\n📦 **Credits:** {clean(credits) or 'not recorded'}"
    if intent == "nqf":       return f"{label}\n\n📊 **NQF Level:** {clean(nqf) or 'not recorded'}"
    if intent == "aim":
        val = clean(aim)
        return f"{label}\n\n🎯 **Aim/Purpose:**\n{val or 'not recorded'}"
    if intent == "outcomes":
        val = clean(content)
        return f"{label}\n\n✅ **Outcomes/Content:**\n{val[:600] if val else 'not recorded'}"
    if intent == "admission":
        if admission and isinstance(admission, list):
            items = "\n".join(f"- {a}" for a in admission if str(a).strip())
            return f"{label}\n\n📋 **Admission Requirements:**\n{items}"
        return f"{label}\n\n📋 **Admission Requirements:** {clean(admission) or 'not recorded'}"
    if intent == "duration":  return f"{label}\n\n⏱️ **Duration:** {clean(duration) + ' year(s)' if clean(duration) else 'not recorded'}"
    if intent == "aps":       return f"{label}\n\n📝 **Minimum APS:** {clean(aps) or 'not recorded'}"
    if intent == "faculty":   return f"{label}\n\n🏛️ **Faculty/Department:** {clean(faculty) or clean(dept) or 'not recorded'}"
    if intent == "intake":    return f"{label}\n\n📅 **Intake:** {clean(intake) or 'not recorded'}"
    if intent == "prereq":    return f"{label}\n\n🔗 **Prerequisites:** {clean(prereq) or 'none recorded'}"

    lines = [f"### {'📘' if not is_prog else '🎓'} {code} — {title}"]
    if clean(credits):  lines.append(f"**Credits:** {credits}")
    if clean(nqf):      lines.append(f"**NQF Level:** {nqf}")
    if clean(faculty):  lines.append(f"**Faculty:** {faculty}")
    if clean(dept):     lines.append(f"**Department:** {dept}")
    if clean(duration): lines.append(f"**Duration:** {duration} year(s)")
    if clean(aps):      lines.append(f"**Min APS:** {aps}")
    if clean(degree):   lines.append(f"**Degree:** {degree}")
    if clean(aim):      lines.append(f"\n**Aim:**\n{str(aim)[:400]}")
    if admission and isinstance(admission, list):
        lines.append("\n**Admission Requirements:**")
        lines.extend(f"- {a}" for a in admission[:3] if str(a).strip())
    if clean(prereq):   lines.append(f"\n**Prerequisites:** {prereq}")
    return "\n".join(lines)

def answer_query(query, code_index, faiss_index, chunks, metas, embedder):
    intent = detect_intent(query)
    debug  = [f"🔎 Intent: **{intent}**"]
    match  = CODE_RE.search(query)
    if match:
        raw = match.group(1).upper()
        doc = code_index.get(raw) or code_index.get(raw.split("/")[0])
        if doc:
            is_prog = "admission_requirements" in doc or "duration_years" in doc
            debug.append(f"✅ Code match: `{raw}`")
            return build_answer(doc, intent, is_prog), "\n\n".join(debug)
    q_lower = query.lower()
    for meta in metas:
        doc = meta["doc"]
        t   = (doc.get("title") or doc.get("name") or "").lower()
        if t and len(t) > 4 and t in q_lower:
            debug.append("✅ Title match")
            return build_answer(doc, intent, meta["type"]=="programme"), "\n\n".join(debug)
    debug.append("🔄 Semantic search")
    emb = embedder.encode([query], convert_to_numpy=True)
    _, indices = faiss_index.search(emb, 1)
    if indices[0][0] < len(chunks):
        meta = metas[indices[0][0]]
        return build_answer(meta["doc"], intent, meta["type"]=="programme"), "\n\n".join(debug)
    return "😕 No result found. Try rephrasing or include a code.", "\n\n".join(debug)

st.set_page_config(page_title="UniZulu Handbook QA", page_icon="📘", layout="wide")
st.title("📘 University of Zululand — Handbook Q&A")
st.caption("Ask in **English** or **isiZulu** about modules, programmes, credits, NQF, admission requirements, and more.")

with st.spinner("Loading..."):
    ner_tok, ner_model  = load_ner_model()
    embedder            = load_embedder()
    modules, programmes = load_data()
    code_index, chunks, metas = build_indexes(modules, programmes)
    faiss_index         = build_faiss(embedder, chunks)

st.success(f"✅ {len(modules)} modules | {len(programmes)} programmes | {len(code_index)} codes indexed")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("XLM-RoBERTa Model + FAISS retrieval")
    st.markdown("""**Examples:**
- *What are the credits for 4AAG211?*
- *Ikhodi yemojuli ngu?*
- *Aim of 4HYD321?*
- *Admission requirements for 1DDEG1?*
- *NQF level of Bachelor of Commerce?*
- *Iminyaka yokufunda ye BA Sociology?*""")
    show_debug = st.toggle("Debug", value=False)
    if st.button("🗑️ Clear"): st.session_state.messages = []; st.rerun()

if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask in English or isiZulu..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, debug = answer_query(prompt, code_index, faiss_index, chunks, metas, embedder)
        st.markdown(answer)
        if show_debug:
            with st.expander("🔬 Debug"): st.markdown(debug)
        st.session_state.messages.append({"role":"assistant","content":answer})
