import os
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MicroZip RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# ── Constants ──────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 4

# ── Session State ──────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "qa_system": None,
    "vectorstore": None,
    "doc_loaded": False,
    "doc_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load Embeddings ────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ── Load Document ──────────────────────────────────────────────────────────
def load_document(file):
    suffix = Path(file.name).suffix.lower()
    tmp_path = f"C:/tmp/uploaded_doc{suffix}"

    os.makedirs("C:/tmp", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(file.read())

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        st.error("Unsupported file type")
        return []

    return loader.load()

# ── Build Vector Store ─────────────────────────────────────────────────────
def build_vectorstore(documents, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)

# ── Build QA System ────────────────────────────────────────────────────────
def build_qa_system(vectorstore, groq_api_key):

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
        temperature=0.2
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    return {
        "llm": llm,
        "retriever": retriever
    }

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG Setup")

    groq_key = st.text_input("Groq API Key", type="password")

    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt", "md"])

    if uploaded_file and groq_key:
        if st.button("Build Knowledge Base"):

            with st.spinner("Processing..."):
                embeddings = load_embeddings()
                docs = load_document(uploaded_file)

                if docs:
                    vs = build_vectorstore(docs, embeddings)
                    qa = build_qa_system(vs, groq_key)

                    st.session_state.vectorstore = vs
                    st.session_state.qa_system = qa
                    st.session_state.doc_loaded = True
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.chat_history = []

                    st.success("✅ Ready!")

# ── Main UI ────────────────────────────────────────────────────────────────
st.title("🤖 MicroZip RAG AI Agent")

if not st.session_state.doc_loaded:
    st.info("Upload document + API key to start")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):

        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                qa = st.session_state.qa_system
                retriever = qa["retriever"]
                llm = qa["llm"]

                # ✅ FIXED HERE
                docs = retriever.invoke(prompt)

                context = "\n\n".join([d.page_content for d in docs])

                # Manual memory
                history = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history]
                )

                final_prompt = f"""
You are a helpful assistant. Answer ONLY from the context.
If the answer is not in the context, say: "I couldn't find that in the document."

Context:
{context}

Chat History:
{history}

Question: {prompt}

Answer:
"""

                response = llm.invoke(final_prompt)
                answer = response.content

                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })