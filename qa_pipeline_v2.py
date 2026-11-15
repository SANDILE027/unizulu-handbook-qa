"""
qa_pipeline_v2.py -- improved answer path for the handbook QA system.
Reuses detect_intent / build_answer from streamlit_handbook.py (passed in as `ns`), and changes only
HOW the record is found and WHEN the system refuses:
  1. code in the question: answer it, or REFUSE if the code is not in the handbook (no more guessing);
     the code pattern also accepts codes such as 1DDEG1 (1-3 digits);
  2. full title or programme name (with or without its code): longest match wins;
  3. lexical match: most of a title's content words appear in the question;
  4. semantic search, but only when the nearest record is closer than `tau`; otherwise REFUSE.
An intent with no keyword no longer crashes.
"""
import re

CODE_V2 = re.compile(r"\b([0-9][A-Z]{2,4}[0-9]{1,3}(?:/[A-Z0-9]+)?)\b", re.IGNORECASE)
STOP = set("the of and in to for a an is are what which how many much does do when who where with on at by "
           "programme program module bachelor degree course".split())
DECLINE = "😕 No relevant result found. Try rephrasing or include a module/programme code."


def _norm(s):
    return re.sub(r"[^0-9a-z]+", " ", str(s).lower()).strip()


def _names(doc):
    raw = doc.get("title") or doc.get("name") or ""
    out = {_norm(raw)}
    no_code = re.sub(r"^\s*[0-9][A-Za-z]{2,4}[0-9]{1,3}\s*[-–—:]?\s*", "", raw)
    out.add(_norm(no_code))
    return {n for n in out if len(n) > 4}


def _tokens(s):
    return {w for w in _norm(s).split() if len(w) > 2 and w not in STOP}


# ---- v3 intent detection -------------------------------------------------------------------
# Field words are checked BEFORE the code/title intents, so a question that merely NAMES the code
# ("ikhodi X", "code X") is no longer answered with the code. Whole-word matching, English + isiZulu.
INTENT_V3 = [
    ("aps",       ["aps", "points required", "minimum points"]),
    ("prereq",    ["prerequisite", "prerequisites", "before", "passed", "must have", "izimfanelo zangaphambili",
                   "ngiqale", "kuphase", "required modules"]),
    ("aim",       ["aim", "purpose", "inhloso", "achieve", "learn", "izama ukufundisa", "ukufundisa"]),
    ("credits",   ["credit", "credits", "worth", "points", "ikhredithi", "amakhredithi", "ama khredithi"]),
    ("nqf",       ["nqf", "level", "izinga", "ibanga"]),
    ("outcomes",  ["outcome", "outcomes", "imiphumela", "imiphumelo"]),
    ("admission", ["admission", "requirement", "requirements", "entry", "apply", "get into", "marks", "matric",
                   "qualify", "izimfanelo zokungena", "ukungena", "ngingangena", "amatikula"]),
    ("duration",  ["duration", "years", "how long", "how much time", "spend", "finish", "iminyaka",
                   "isikhathi esingakanani", "ukuqeda"]),
    ("intake",    ["intake", "month", "begin", "begins", "start", "starts", "commence", "semester",
                   "inyanga", "iqala", "ukuqaliswa"]),
    ("faculty",   ["faculty", "department", "taught", "belong", "belongs", "under", "runs", "run by", "offers",
                   "offered", "umnyango", "ifakhalthi", "ifundiswa", "umphathi", "luphethwe"]),
    ("lecturer",  ["lecturer", "who teaches", "umfundisi", "instructor", "professor"]),
    ("code",      ["code", "ikhodi"]),
    ("title",     ["title", "isihloko", "name", "igama", "called"]),
]


def detect_intent_v3(query):
    q = query.lower()
    for field, kws in INTENT_V3:
        if any(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", q) for k in kws):
            return field
    return "general"


def answer_query_v2(query, code_index, faiss_index, chunks, metas, embedder, ns, tau=None, intent_mode="app"):
    try:
        intent = detect_intent_v3(query) if intent_mode == "v3" else ns["detect_intent"](query)
    except Exception:
        intent = "general"
    debug = [f"🔎 Intent: **{intent}**"]

    # a code is any word of the question that is a known code (also letter-first codes such as SNPN312),
    # or, failing that, a code-shaped word that is NOT in the handbook
    words = [w.upper() for w in re.findall(r"[0-9A-Za-z/]+", query)]
    known = next((w for w in words if w in code_index or w.split("/")[0] in code_index), None)
    m = CODE_V2.search(query)
    if known or m:
        raw = known or m.group(1).upper()
        doc = code_index.get(raw) or code_index.get(raw.split("/")[0])
        if doc:
            debug.append(f"🔑 Code found: `{raw}`\n\n✅ **Exact code match**")
            return ns["build_answer"](doc, intent, "admission_requirements" in doc or "duration_years" in doc), "\n\n".join(debug)
        debug.append(f"⚠️ Code `{raw}` not in dataset → refuse")
        return DECLINE, "\n\n".join(debug)

    qn = " " + _norm(query) + " "
    best = None
    for meta in metas:
        for n in _names(meta["doc"]):
            if f" {n} " in qn and (best is None or len(n) > best[0]):
                best = (len(n), meta)
    if best:
        debug.append("✅ **Title match**")
        return ns["build_answer"](best[1]["doc"], intent, best[1]["type"] == "programme"), "\n\n".join(debug)

    qt = _tokens(query)
    cands = []
    for meta in metas:
        tt = _tokens(meta["doc"].get("title") or meta["doc"].get("name") or "")
        if len(tt) >= 2:
            cov = len(tt & qt) / len(tt)
            if cov >= 0.75:
                cands.append((cov, len(tt), meta))
    if cands:
        cands.sort(key=lambda c: (-c[0], -c[1]))
        if len(cands) == 1 or (cands[0][0], cands[0][1]) != (cands[1][0], cands[1][1]):
            meta = cands[0][2]
            debug.append("✅ **Lexical match**")
            return ns["build_answer"](meta["doc"], intent, meta["type"] == "programme"), "\n\n".join(debug)

    emb = embedder.encode([query], convert_to_numpy=True)
    d, i = faiss_index.search(emb, 3)
    dist = float(d[0][0])
    debug.append(f"🔄 **Semantic search (FAISS)**\n\n📌 distance {dist:.2f}")
    meta = metas[int(i[0][0])]
    if tau is None or dist > tau:  # default: do not answer from a loose semantic match, only suggest it
        near = (meta["doc"].get("title") or meta["doc"].get("name") or "?")[:80]
        debug.append("→ no confident match, refuse")
        return DECLINE + f" (Closest handbook entry: {near}.)", "\n\n".join(debug)
    return ns["build_answer"](meta["doc"], intent, meta["type"] == "programme"), "\n\n".join(debug)