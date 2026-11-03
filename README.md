# 📚 AI PDF Chatbot (RAG Application)

A production-ready **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions based on their content. The AI strictly answers using the provided document context, eliminating hallucinations.

## 🔗 Live Demo
👉 **[Click here to view the Live App](https://rag-pdf-chatbot-01.streamlit.app/)**

## ✨ Key Features
- **Document Ingestion:** Upload single or multiple PDF files.
- **Smart Chunking:** Splits text using `RecursiveCharacterTextSplitter` for optimal context.
- **Advanced Retrieval:** Uses **MMR (Maximum Marginal Relevance)** search to fetch diverse and relevant chunks, avoiding repetitive information.
- **Source Citations:** Displays exact page numbers and text snippets from the PDF to verify the AI's answer.
- **Smart Caching:** Uses MD5 hashing to prevent re-embedding previously uploaded documents, saving time and server costs.
- **Zero Hallucination:** Strict system prompt ensures the AI only answers from the provided PDF.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **LLM:** Groq API (Llama 3)
- **Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chaudhary-Momna/RAG-PDF-Chatbot.git
   cd RAG-PDF-Chatbot

2.Create a virtual environment & install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3.Set up Environment Variables:
Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY="your_groq_api_key_here"

4.Run the application:
streamlit run app.py

👨‍💻 Author
Chaudhary Momna

GitHub: @Chaudhary-Momna

LinkedIn: Chaudhary Momna

