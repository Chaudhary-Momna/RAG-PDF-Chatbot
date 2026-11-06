import streamlit as st
import os
import tempfile
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
st.title("📚 AI Study Buddy (PDF Chat)")
st.write("Upload your textbook or notes and customize how the AI processes it.")

# ---------------- Sidebar (Settings & Upload) ---------------- #
with st.sidebar:
    st.header("📄 Your Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    st.subheader("⚙️ RAG Settings")
    
    chunk_size = st.slider("Chunk Size (Characters)", min_value=500, max_value=2500, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    
    
    embedding_choice = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
    )
    
    st.info("Welcome! Upload your textbook and I will help you study from it.")
    
    if st.button("🗑️ Clear Chat & Reset"):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.current_hash = None
        st.rerun()
        
    # Document Summary Button
    if st.session_state.get("retriever") is not None:
        st.subheader("⚡ Quick Actions")
        if st.button("Generate Document Summary"):
            with st.spinner("Summarizing document..."):
                docs = st.session_state.retriever.invoke("Give me a detailed summary of this document.")
                context = "\n\n".join([doc.page_content for doc in docs])
                summary_prompt = f"Please provide a concise but detailed summary of the following text:\n\n{context}"
                
                llm = ChatGroq(model="openai/gpt-oss-120b")
                summary = llm.invoke(summary_prompt)
                st.info(summary.content)

# ---------------- Initialization ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_hash" not in st.session_state:
    st.session_state.current_hash = None

# ---------------- Process PDF ---------------- #
def process_pdf(file, chunk_size, chunk_overlap, embedding_model_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Check if PDF is scanned
    total_text = "".join([doc.page_content for doc in docs])
    if len(total_text.strip()) < 50:
        os.remove(tmp_path)
        return None, "This PDF seems to be scanned (images only). Please upload a text-based PDF."

    # Apply custom chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)

    # Apply custom Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    os.remove(tmp_path)
    
    # Return retriever with k=4 (You can make this customizable too if you want)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2}), None

# ---------------- Main Chat Interface ---------------- #
if uploaded_file is not None:
    # Create a unique hash based on file + settings so it reprocesses if user changes settings
    hash_input = uploaded_file.getvalue() + str(chunk_size).encode() + str(chunk_overlap).encode() + embedding_choice.encode()
    file_hash = hashlib.md5(hash_input).hexdigest()
    
    # If new file or settings changed, reprocess
    if st.session_state.current_hash != file_hash:
        with st.spinner("Processing PDF with your custom settings... (This may take a minute)"):
            retriever, error = process_pdf(uploaded_file, chunk_size, chunk_overlap, embedding_choice)
            if error:
                st.error(error)
                st.stop()
            else:
                st.session_state.retriever = retriever
                st.session_state.messages = [] # Clear chat on new upload
                st.session_state.current_hash = file_hash
                st.success("PDF Processed Successfully! You can ask questions now.")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if query := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Retrieve Context
                docs = st.session_state.retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # 2. Create Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI Study Buddy. Use ONLY the provided context to answer. If the answer is not in the context, say 'I could not find this in your book.' Keep answers educational."),
                    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
                ])
                
                final_prompt = prompt.invoke({"context": context, "question": query})
                
                # 3. Call LLM with Streaming
                llm = ChatGroq(model="openai/gpt-oss-120b")
                response_text = st.write_stream(llm.stream(final_prompt))
                
                # 4. Show Sources (Citations)
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(docs):
                        page_num = doc.metadata.get("page", 0) + 1
                        st.write(f"**Source {i+1} (Page {page_num}):**")
                        st.info(doc.page_content[:200] + "...") 

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    st.info("👈 Please upload a PDF file from the sidebar to begin.")
