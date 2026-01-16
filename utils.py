import os
import re
import uuid
import tempfile
from dataclasses import dataclass
from typing import List

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from transformers import pipeline

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

@dataclass
class ResearchPaper:
    paper_id: str
    source: str
    abstract: str
    full_text: str

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_abstract(text: str) -> str:
    lower = text.lower()
    if "abstract" not in lower:
        return ""
    start = lower.find("abstract")
    slice_text = text[start:start + 2500]
    stop_keywords = ["keywords", "index terms", "introduction", "1.", "\n\n"]
    for kw in stop_keywords:
        idx = slice_text.lower().find(kw)
        if idx != -1 and idx > 200:
            slice_text = slice_text[:idx]
            break
    return clean_text(slice_text.replace("Abstract", "").replace("ABSTRACT", ""))

def is_noise(text: str) -> bool:
    noise_terms = ["received the", "degree from", "university", "acknowledgment", "references", "editor", "©", "ieee"]
    t = text.lower()
    return any(n in t for n in noise_terms)

def load_document(uploaded_file=None, url=None) -> str:
    docs = []
    if uploaded_file:
        suffix = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name
        docs = PyPDFLoader(path).load() if suffix == "pdf" else Docx2txtLoader(path).load()
    elif url:
        docs = WebBaseLoader(url).load()
    return clean_text(" ".join(d.page_content for d in docs))

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, temperature=0.2)

def build_vector_store(papers: List[ResearchPaper]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=100)
    chunks = []
    for p in papers:
        if p.abstract: chunks.append(p.abstract)
        body_chunks = splitter.split_text(p.full_text)
        chunks.extend([c for c in body_chunks if not is_noise(c)])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

def answer_question(question: str, vectorstore: FAISS) -> str:
    docs = vectorstore.similarity_search(question, k=12)
    context_chunks = [d.page_content for d in docs if not is_noise(d.page_content)][:5]
    if not context_chunks: return "The answer is not present in the provided context."
    
    context = "\n\n".join(context_chunks)
    prompt = f"Answer the question using ONLY the context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    llm = load_llm()
    return llm(prompt)[0]["generated_text"].strip()