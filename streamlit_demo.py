#!/usr/bin/env python3
"""
streamlit_demo.py -- live demonstration of the three QA versions evaluated in the dissertation.

  Original  = streamlit_handbook.py answer path (never refuses)
  Version 2 = qa_pipeline_v2.answer_query_v2, intent_mode="app"  (refusal, longest title, lexical match)
  Version 3 = qa_pipeline_v2.answer_query_v2, intent_mode="v3"   (field words before code/title, EN + ZU)

It builds the indexes exactly as run_qa_eval.py does, so the answers match the evaluated runs.
Put this file in the same folder as streamlit_handbook.py, qa_pipeline_v2.py and the two
normalized_*.json files, then:   streamlit run streamlit_demo.py
"""
import ast
import inspect
import json
import os
import re
import sys
from collections import Counter

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(HERE, "streamlit_handbook.py")
BASE_DIR = HERE
DEFAULT_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

sys.path.insert(0, HERE)


# ---- same loader as run_qa_eval.py: runs the app's functions/constants, not its UI --------------
def load_pipeline(app_path, base_dir):
    src = open(app_path, encoding="utf-8").read()
    tree = ast.parse(src)
    skip_funcs = {"load_ner_model", "load_embedder", "load_data"}
    import numpy as np
    import faiss
    ns = {"re": re, "json": json, "np": np, "numpy": np, "os": os, "faiss": faiss,
          "Counter": Counter, "BASE_DIR": base_dir}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names]
            if any(n in ("streamlit", "torch", "transformers", "sentence_transformers", "pandas") for n in names):
                continue
        elif isinstance(node, ast.FunctionDef):
            if node.name in skip_funcs:
                continue
            node.decorator_list = []
        elif not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        try:
            m = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(m)
            exec(compile(m, app_path, "exec"), ns)
        except Exception:
            pass
    return ns


@st.cache_resource(show_spinner="Loading the handbook and the embedding model (first time only)...")
def load_everything():
    ns = load_pipeline(APP_PATH, BASE_DIR)
    mod_file = ns.get("MOD_FILE") or os.path.join(BASE_DIR, "normalized_modules.json")
    prog_file = ns.get("PROG_FILE") or os.path.join(BASE_DIR, "normalized_programmes.json")
    modules = json.load(open(mod_file, encoding="utf-8"))
    programmes = json.load(open(prog_file, encoding="utf-8"))
    code_index, chunks, metas = ns["build_indexes"](modules, programmes)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(ns.get("EMBED_MODEL") or DEFAULT_EMBED)
    index = ns["build_faiss"](embedder, chunks)
    return ns, code_index, index, chunks, metas, embedder, len(modules), len(programmes)


def run_original(query, ns, code_index, index, chunks, metas, embedder):
    """Call the app's own answer_query, filling its parameters by name, as the evaluation did."""
    fn = ns["answer_query"]
    pool = {"query": query, "q": query, "prompt": query, "question": query,
            "code_index": code_index, "index": index, "faiss_index": index,
            "chunks": chunks, "metas": metas, "embedder": embedder, "model": embedder, "ns": ns}
    kwargs = {}
    for name, p in inspect.signature(fn).parameters.items():
        if name in pool:
            kwargs[name] = pool[name]
        elif p.default is inspect._empty:
            raise TypeError(f"answer_query needs an argument '{name}' that the demo does not know")
    return fn(**kwargs)


def answer(version, query, bundle):
    ns, code_index, index, chunks, metas, embedder, _, _ = bundle
    if version.startswith("Original"):
        out = run_original(query, ns, code_index, index, chunks, metas, embedder)
    else:
        from qa_pipeline_v2 import answer_query_v2
        mode = "v3" if version.startswith("Version 3") else "app"
        out = answer_query_v2(query, code_index, index, chunks, metas, embedder, ns, None, mode)
    if isinstance(out, tuple):
        resp, debug = out[0], out[1] if len(out) > 1 else ""
    else:
        resp, debug = out, ""
    return resp, debug


# ---- page ------------------------------------------------------------------------------------------
st.set_page_config(page_title="UniZulu Module Guide QA", page_icon="📘", layout="wide")
st.title("📘 UniZulu Module Guide QA")
st.caption("Ask about a module or programme in English or isiZulu: credits, NQF level, prerequisites, "
           "aim, admission, duration, intake, faculty.")

try:
    bundle = load_everything()
except Exception as e:
    st.error(f"Could not load the pipeline: {e}")
    st.stop()

with st.sidebar:
    st.header("Version")
    version = st.radio(
        "Which version answers?",
        ["Version 3 (final)", "Version 2", "Original (before changes)"],
        index=0,
        help="Original = the first application, which never refuses. Versions 2 and 3 are the improved "
             "pipelines evaluated on the 154-query blind set.")
    st.markdown(
        "- **Original**: always returns the closest record.\n"
        "- **Version 2**: refuses when the code or title is not in the handbook.\n"
        "- **Version 3**: version 2 plus English and isiZulu field words checked first.")
    st.divider()
    st.metric("Modules in the handbook", bundle[6])
    st.metric("Programmes in the handbook", bundle[7])
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

EXAMPLES = [
    "What are the credits for 1RTO311?",
    "Mangaki amakhredithi emojuli 1RTO311?",
    "What are the prerequisites for 1RTO311?",
    "What is the NQF level of 1RTO311?",
    "What are the credits for ZZZ999?",
    "What is the weather today?",
]
if "history" not in st.session_state:
    st.session_state.history = []

st.write("**Try an example, or type your own question below:**")
cols = st.columns(3)
clicked = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        clicked = ex

typed = st.chat_input("Type a question in English or isiZulu...")
query = typed or clicked
if query:
    try:
        resp, debug = answer(version, query, bundle)
    except Exception as e:
        resp, debug = f"Sorry, something went wrong: {e}", ""
    st.session_state.history.append({"q": query, "a": resp, "d": debug, "v": version})

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])
        st.caption(f"Answered by: {turn['v']}")
        if turn["d"]:
            with st.expander("How the system decided"):
                st.markdown(turn["d"])