import streamlit as st
import uuid
from utils import load_document, extract_abstract, build_vector_store, answer_question, ResearchPaper

def render_sidebar():
    st.sidebar.header("📄 Ingest Research Paper")
    uploaded = st.sidebar.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
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

    if st.sidebar.button("Build Knowledge Index"):
        if st.session_state.papers:
            st.session_state.vectorstore = build_vector_store(st.session_state.papers)
            st.success("FAISS knowledge index built")
        else:
            st.sidebar.error("Please ingest papers first.")

def render_main_content():
    st.header("📝 Abstract Summary")
    if st.session_state.papers:
        for p in st.session_state.papers:
            with st.expander(p.source):
                st.write(p.abstract if p.abstract else "Abstract not detected.")

    st.header("🔍 Semantic Search")
    query = st.text_input("Search research topics")
    if query and st.session_state.vectorstore:
        results = st.session_state.vectorstore.similarity_search(query, k=5)
        for r in results:
            st.markdown(r.page_content[:500])

    st.header("💬 Research Q&A")
    question = st.text_input("Ask a research question")
    if question and st.session_state.vectorstore:
        answer = answer_question(question, st.session_state.vectorstore)
        st.success(answer)

    st.header("📖 Paper Library")
    for p in st.session_state.papers:
        with st.expander(p.source):
            st.write(f"**ABSTRACT:**\n{p.abstract or 'Not available'}")
            st.divider()
            st.write(f"**FULL TEXT (PREVIEW):**\n{p.full_text[:2000]}")