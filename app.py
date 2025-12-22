# ============================================================
# GA03 – Research Paper Management & Analysis Intelligence
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import uuid
import re
import tempfile
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader
)

from transformers import pipeline

# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class ResearchPaper:
    paper_id: str
    source: str
    abstract: str
    full_text: str

# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_abstract(text: str) -> str:
    """
    IEEE-style abstract extraction
    """
    lower = text.lower()

    if "abstract" not in lower:
        return ""

    start = lower.find("abstract")
    slice_text = text[start:start + 2500]

    stop_keywords = [
        "keywords", "index terms", "introduction", "1.", "\n\n"
    ]

    for kw in stop_keywords:
        idx = slice_text.lower().find(kw)
        if idx != -1 and idx > 200:
            slice_text = slice_text[:idx]
            break

    return clean_text(slice_text.replace("Abstract", "").replace("ABSTRACT", ""))

def is_noise(text: str) -> bool:
    noise_terms = [
        "received the", "degree from", "university",
        "acknowledgment", "references",
        "editor", "committee", "biography",
        "contact him at", "©", "ieee"
    ]
    t = text.lower()
    return any(n in t for n in noise_terms)

# ============================================================
# DOCUMENT INGESTION
# ============================================================

def load_document(uploaded_file=None, url=None) -> str:
    docs = []

    if uploaded_file:
        suffix = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        if suffix == "pdf":
            docs = PyPDFLoader(path).load()
        else:
            docs = Docx2txtLoader(path).load()

    elif url:
        docs = WebBaseLoader(url).load()

    full_text = " ".join(d.page_content for d in docs)
    return clean_text(full_text)

# ============================================================
# LLM
# ============================================================

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.2
    )

# ============================================================
# RAG QA
# ============================================================

def answer_question(question: str, vectorstore: FAISS) -> str:
    docs = vectorstore.similarity_search(question, k=12)

    context_chunks = [
        d.page_content for d in docs
        if not is_noise(d.page_content)
    ][:5]

    if not context_chunks:
        return "The answer is not present in the provided context."

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an academic research assistant.

Task:
Answer the question using ONLY the context.

Rules:
- Do not quote biographies or references
- Be factual and concise
- If the answer is missing, say so clearly
- If the question asks for a summary, produce a structured summary in 4–6 sentences

Context:
{context}

Question:
{question}

Answer:
"""

    llm = load_llm()
    return llm(prompt)[0]["generated_text"].strip()

# ============================================================
# STREAMLIT UI (UNCHANGED)
# ============================================================

st.set_page_config("GA03 Research Intelligence", layout="wide")
st.title("📚 GA03 – Research Paper Management & Analysis Intelligence System")

if "papers" not in st.session_state:
    st.session_state.papers: List[ResearchPaper] = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------- SIDEBAR ----------------
st.sidebar.header("📄 Ingest Research Paper")

uploaded = st.sidebar.file_uploader(
    "Upload PDF or DOCX", type=["pdf", "docx"]
)

url = st.sidebar.text_input("Or paste paper URL")

if st.sidebar.button("Ingest Paper"):
    full_text = load_document(uploaded, url)
    abstract = extract_abstract(full_text)

    paper = ResearchPaper(
        paper_id=str(uuid.uuid4()),
        source=uploaded.name if uploaded else url,
        abstract=abstract,
        full_text=full_text
    )

    st.session_state.papers.append(paper)
    st.success("Paper ingested successfully")

# ---------------- BUILD INDEX ----------------
if st.sidebar.button("Build Knowledge Index"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=100
    )

    chunks = []

    for p in st.session_state.papers:
        if p.abstract:
            chunks.append(p.abstract)
        body_chunks = splitter.split_text(p.full_text)
        chunks.extend([c for c in body_chunks if not is_noise(c)])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
    st.success("FAISS knowledge index built")

# ---------------- ABSTRACT SUMMARY ----------------
st.header("📝 Abstract Summary")

if st.session_state.papers:
    for p in st.session_state.papers:
        with st.expander(p.source):
            st.write(p.abstract if p.abstract else "Abstract not detected.")

# ---------------- SEMANTIC SEARCH ----------------
st.header("🔍 Semantic Search")

query = st.text_input("Search research topics")

if query and st.session_state.vectorstore:
    results = st.session_state.vectorstore.similarity_search(query, k=5)
    for r in results:
        st.markdown(r.page_content[:500])

# ---------------- RAG Q&A ----------------
st.header("💬 Research Q&A")

question = st.text_input("Ask a research question")

if question and st.session_state.vectorstore:
    answer = answer_question(question, st.session_state.vectorstore)
    st.success(answer)

# ---------------- PAPER LIBRARY ----------------
st.header("📖 Paper Library")

for p in st.session_state.papers:
    with st.expander(p.source):
        st.write("ABSTRACT:")
        st.write(p.abstract if p.abstract else "Not available")
        st.divider()
        st.write("FULL TEXT (PREVIEW):")
        st.write(p.full_text[:2000])
