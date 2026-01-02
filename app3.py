import os
import uuid
import re
import tempfile
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# 1. RESEARCH DATA MODELS (Requirements Part I)
# ============================================================

class PaperSection(BaseModel):
    name: str
    content: str

class ResearchPaper(BaseModel):
    paper_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: List[str] = []
    year: int = 2024
    abstract: str
    sections: List[PaperSection] = []
    keywords: List[str] = []
    source: str

# ============================================================
# 2. ADVANCED PARSING & METADATA (Requirements Part I & V)
# ============================================================

def advanced_parse_pdf(file_path, original_name) -> ResearchPaper:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])
    
    # Logic for Metadata Extraction
    lines = full_text.split('\n')
    title = " ".join(lines[:2]).strip() if len(lines) > 2 else original_name
    
    # Optimization 2: Year & Keyword Extraction
    year_match = re.search(r"20\d{2}", full_text[:3000])
    extracted_year = int(year_match.group(0)) if year_match else 2024
    
    kw_match = re.search(r"(?i)(keywords|index terms):?\s*(.*)", full_text[:5000])
    keywords = [k.strip() for k in kw_match.group(2).split(',')[:5]] if kw_match else ["General Research"]

    # Section-Level Extraction
    section_patterns = {
        "Abstract": r"(?i)abstract",
        "Introduction": r"(?i)1\.?\s+introduction|introduction",
        "Methods": r"(?i)methodology|proposed method|materials",
        "Results": r"(?i)results|experiments",
        "Conclusion": r"(?i)conclusion|summary",
        "References": r"(?i)references"
    }
    
    found_sections = []
    current_section = "Header/Title"
    current_content = []
    
    for line in lines:
        for sec_name, pattern in section_patterns.items():
            if re.match(pattern, line.strip()):
                found_sections.append(PaperSection(name=current_section, content="\n".join(current_content)))
                current_section = sec_name
                current_content = []
                break
        current_content.append(line)
    
    found_sections.append(PaperSection(name=current_section, content="\n".join(current_content)))
    
    abstract_content = next((s.content for s in found_sections if "Abstract" in s.name), "No Abstract Detected")

    return ResearchPaper(
        title=title,
        year=extracted_year,
        abstract=abstract_content,
        sections=found_sections,
        keywords=keywords,
        source=original_name
    )

# ============================================================
# 3. VECTOR STORAGE (Requirements Part II)
# ============================================================

@st.cache_resource
def get_embedding_model():
    # Uses the model compatible with sentence-transformers
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def index_papers(papers: List[ResearchPaper]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    documents = []
    metadatas = []
    
    for paper in papers:
        for section in paper.sections:
            chunks = text_splitter.split_text(section.content)
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append({
                    "title": paper.title,
                    "section": section.name,
                    "year": paper.year,
                    "id": paper.paper_id
                })
    
    return FAISS.from_texts(documents, get_embedding_model(), metadatas=metadatas)

# ============================================================
# 4. INTELLIGENT RAG ENGINE (Requirements Part III)
# ============================================================

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)

def generate_research_answer(query: str, vectorstore):
    # Retrieve top 6 chunks for broader context
    docs = vectorstore.similarity_search(query, k=6)
    
    context_entries = []
    for d in docs:
        context_entries.append(f"Source: {d.metadata['title']} | Section: {d.metadata['section']}\nContent: {d.page_content}")
    
    context = "\n\n".join(context_entries)
    
    prompt = f"""
    You are a Research Intelligence AI. Use the provided context to answer the question accurately.
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    Detailed Academic Answer:"""
    
    llm = load_llm()
    result = llm(prompt, max_new_tokens=300, do_sample=False)
    return result[0]['generated_text'], docs

# ============================================================
# 5. STREAMLIT UI (Requirements Part VI)
# ============================================================

st.set_page_config(page_title="InsightResearch Intelligence", layout="wide", page_icon="📚")

# Initialize Session States
if "library" not in st.session_state:
    st.session_state.library = []
if "vs" not in st.session_state:
    st.session_state.vs = None

st.title("📚 Research Paper Management & Analysis System")
st.markdown("---")

# SIDEBAR: Ingestion
with st.sidebar:
    st.header("📥 Ingest Content")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Process & Index"):
        if uploaded_files:
            with st.spinner("Analyzing papers..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        paper_obj = advanced_parse_pdf(tmp.name, uploaded_file.name)
                        st.session_state.library.append(paper_obj)
                
                st.session_state.vs = index_papers(st.session_state.library)
                st.success(f"Successfully indexed {len(uploaded_files)} papers.")
        else:
            st.warning("Please upload files first.")

# MAIN INTERFACE TABS
tab1, tab2, tab3 = st.tabs(["📖 Library Explorer", "💬 Research Assistant", "📊 Analytics & Trends"])

with tab1:
    if not st.session_state.library:
        st.info("Your library is empty. Upload papers via the sidebar to begin.")
    else:
        for paper in st.session_state.library:
            with st.expander(f"📄 {paper.title} ({paper.year})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Abstract:**")
                    st.caption(paper.abstract)
                with col2:
                    st.write("**Metadata:**")
                    st.write(f"🏷️ Keywords: {', '.join(paper.keywords)}")
                    st.write(f"📂 Sections: {len(paper.sections)}")

with tab2:
    st.header("Interact with your Research")
    query = st.text_input("Ask a question across your library (e.g., 'What are the main limitations of the proposed methods?')")
    
    if query:
        if st.session_state.vs:
            with st.spinner("Consulting library..."):
                answer, sources = generate_research_answer(query, st.session_state.vs)
                st.markdown(f"### 🤖 Answer\n{answer}")
                
                st.markdown("---")
                st.subheader("📍 Source References")
                for s in sources:
                    st.info(f"**Paper:** {s.metadata['title']} | **Section:** {s.metadata['section']}")
        else:
            st.error("Please build the knowledge index first.")

with tab3:
    st.header("Research Trend Identification")
    if st.session_state.library:
        # Create Dataframe for Analytics
        df = pd.DataFrame([
            {"Year": p.year, "Title": p.title, "Keywords": ", ".join(p.keywords)} 
            for p in st.session_state.library
        ])
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Publication Year Trend")
            st.bar_chart(df['Year'].value_counts())
        
        with c2:
            st.subheader("Key Topic Distribution")
            # Simple keyword frequency
            all_kw = []
            for p in st.session_state.library: all_kw.extend(p.keywords)
            kw_counts = pd.Series(all_kw).value_counts()
            st.dataframe(kw_counts, column_config={"_index": "Topic", "value": "Count"})
    else:
        st.info("Upload papers to see trend analysis.")