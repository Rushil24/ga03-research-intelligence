# 📚 GA03 – Research Paper Management & Analysis Intelligence System

An end-to-end Research Paper Ingestion, Semantic Search, and RAG-based Question Answering System built using Python, LangChain, FAISS, HuggingFace Transformers, and Streamlit.
This project is designed to intelligently process academic research papers (PDF/DOCX/URLs), extract meaningful representations, and enable context-aware question answering in an IEEE-compliant manner.

## 🚀 Project Overview

Modern research involves reading, understanding, and synthesizing information from multiple academic papers—an increasingly time-consuming task.
GA03 – Research Intelligence System solves this problem by:

Ingesting research papers from multiple sources

Extracting and cleaning structured textual content

Building a semantic knowledge index using embeddings

Enabling semantic search across papers

Providing retrieval-augmented generation (RAG) based question answering

The system is especially suited for academic research, literature reviews, and technical analysis.

## 🎯 Key Features

📄 Multi-source ingestion

Upload PDF or DOCX research papers

Ingest papers directly via URLs

🧠 IEEE-aware abstract extraction

✂️ Smart text chunking with overlap

🔍 Semantic search using FAISS

💬 RAG-based Question Answering

Answers generated strictly from retrieved context

Avoids hallucinations and irrelevant content

🖥️ Interactive Streamlit UI

⚡ Optimized for CPU execution (no GPU required)

## 🧩 System Architecture

User Input (PDF / DOCX / URL)
        ↓
Document Loader & Cleaner
        ↓
Abstract Extraction (IEEE-style)
        ↓
Text Chunking
        ↓
Embedding Generation (Sentence Transformers)
        ↓
FAISS Vector Store
        ↓
Semantic Search / RAG Q&A (Flan-T5)
        ↓
Streamlit UI Output

## 🛠️ Tech Stack

Python 3.10

Streamlit – Interactive UI

LangChain – Document processing & RAG orchestration

FAISS – Vector similarity search

HuggingFace Transformers – LLM inference

Sentence-Transformers – Embeddings

Flan-T5 (google/flan-t5-base) – Text generation

PyPDF / Docx2Txt / BeautifulSoup – Document loading

## ⚙️ Installation & Setup

1️⃣ Create and activate environment (recommended)

conda create -n ga03 python=3.10 -y

conda activate ga03

2️⃣ Install dependencies

pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

pip install transformers==4.36.2 accelerate==0.25.0

pip install huggingface-hub==0.19.4

pip install langchain==0.1.16

pip install langchain-community==0.0.36

pip install langchain-text-splitters==0.0.1

pip install faiss-cpu==1.7.4

pip install sentence-transformers==2.2.2

pip install streamlit==1.31.1

pip install pypdf docx2txt beautifulsoup4 requests

## ▶️ Running the Application

Navigate to the project directory and run:

streamlit run app.py


The app will be available at:

http://localhost:8501

## 🧪 How to Use

Upload a PDF/DOCX or paste a research paper URL

Click Ingest Paper

Click Build Knowledge Index

Explore:

📝 Extracted abstracts

🔍 Semantic search

💬 Research Q&A based on paper content

📌 Design Decisions

FAISS chosen for fast and scalable similarity search

Flan-T5 selected for instruction-following and factual summarization

Context-restricted prompting used to prevent hallucinations

Noise filtering removes biographies, acknowledgments, and references

Modular design for easy future extension

## 🔮 Future Enhancements

Multi-document citation tracking

Cross-paper comparison & clustering

Export answers with references

User authentication and saved sessions

Cloud deployment (AWS / GCP)

GPU support for large-scale corpora

## 🧠 Learning Outcomes

Practical implementation of RAG pipelines

Hands-on experience with LangChain modular architecture

Understanding real-world LLM + Vector DB integration

Building production-style Streamlit applications

Debugging dependency and environment conflicts in ML systems

## 📽️ Demo & Submission Notes

Project includes:

Paper ingestion

Knowledge indexing

Semantic search

RAG-based Q&A

Suitable for:

Academic evaluation

Internship / job interviews

Research assistant roles

## 👤 Author

Rushil Pajni
B.Tech (Cyber Security)
Python | AI | Data | Research Systems
