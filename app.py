import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon="📚", layout="wide")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document.\"""",
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return HuggingFaceEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b")


# ---------- database building ----------
def build_vectorstore(uploaded_files, chunk_size, chunk_overlap):
    """Create (or reuse) a Chroma DB for the uploaded PDFs."""
    # Hash file contents + settings so the same upload isn't re-embedded
    h = hashlib.md5()
    for f in uploaded_files:
        h.update(f.getvalue())
    h.update(f"{chunk_size}-{chunk_overlap}".encode())
    persist_dir = os.path.join("chroma-db", h.hexdigest())

    embedding_model = get_embedding_model()

    if os.path.exists(persist_dir):  # already processed before
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    all_docs = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            for p in pages:
                p.metadata["source"] = f.name  # show real file name, not temp path
            all_docs.extend(pages)
        finally:
            os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ---------- sidebar ----------
with st.sidebar:
    st.header("📄 Your documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True
    )

    with st.expander("⚙️ Settings"):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
        k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading, chunking and embedding... this can take a minute"):
            vs = build_vectorstore(uploaded_files, chunk_size, chunk_overlap)
            st.session_state.retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": max(10, k * 3), "lambda_mult": 0.5},
            )
            st.session_state.doc_names = [f.name for f in uploaded_files]
            st.session_state.messages = []  # fresh chat for new docs
        st.success("Ready! Ask your questions.")

    if st.session_state.doc_names:
        st.markdown("**Active:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- main chat ----------
st.title("📚 Chat with your PDF")

if st.session_state.retriever is None:
    st.info("👈 Upload a PDF (or a book) in the sidebar and click **Process documents**.")
    st.stop()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"{s['file']} — page {s['page']}")
                    st.write(s["text"])

# New question
if query := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = st.session_state.retriever.invoke(query)
            context = "\n\n".join(d.page_content for d in docs)
            final_prompt = PROMPT.invoke({"context": context, "question": query})
            answer = get_llm().invoke(final_prompt).content

        st.markdown(answer)

        sources = [
            {
                "file": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", 0) + 1,
                "text": d.page_content[:400] + "...",
            }
            for d in docs
        ]
        with st.expander("Sources"):
            for s in sources:
                st.caption(f"{s['file']} — page {s['page']}")
                st.write(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )