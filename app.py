import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── Constants ──────────────────────────────────────────────────────────────────
CORPUS_PATH = "./hr_docs"   # put your PDFs in this folder when deploying
REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics' "
    "internal policy documents. Your question appears to be outside the scope of "
    "what I can help with. Please contact the HR helpdesk at "
    "hr.helpdesk@zyrodynamics.com for other queries."
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the information provided in the context below.
Be concise, accurate, and professional.
If the context does not contain enough information to answer, say so clearly.
Do not make up information or use outside knowledge.

Context from HR Policy Documents:
{context}

Employee Question: {question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Determine if the following question is related to HR policies,
employee benefits, leave, compensation, workplace conduct, performance, onboarding,
separation, travel expenses, IT security, or other topics covered in an employee handbook.

Question: {question}

Reply with only one word: YES if it is HR-related, NO if it is not.
""")

# ── Load & cache RAG pipeline ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HR policies and building knowledge base...")
def load_rag_pipeline():
    google_api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Load PDFs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Embed + vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_output_tokens=512
    )

    return retriever, llm

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "HR Policy")
        source_name = source.split("/")[-1].replace(".pdf", "").replace("_", " ")
        formatted.append(f"[Source: {source_name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def ask_bot(question, retriever, llm):
    parser = StrOutputParser()

    # Guardrail check
    oos_result = parser.invoke(llm.invoke(OOS_PROMPT.invoke({"question": question})))
    if "NO" in oos_result.strip().upper():
        return {"answer": REFUSAL_MESSAGE, "sources": [], "out_of_scope": True}

    # RAG
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = parser.invoke(llm.invoke(RAG_PROMPT.invoke({"context": context, "question": question})))
    sources = list(set([
        d.metadata.get("source", "").split("/")[-1].replace(".pdf", "").replace("_", " ")
        for d in docs
    ]))
    return {"answer": answer, "sources": sources, "out_of_scope": False}

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Powered by RAG | Ask any HR policy question")
st.divider()

# Load pipeline
try:
    retriever, llm = load_rag_pipeline()
    st.success("✅ HR knowledge base loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm your Zyro Dynamics HR assistant. Ask me anything about our HR policies — leave, compensation, WFH, performance reviews, and more!",
        "sources": []
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.write(f"• {src}")

# Chat input
if prompt := st.chat_input("Ask an HR question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(prompt, retriever, llm)

        st.write(result["answer"])

        if result["sources"]:
            with st.expander("📄 Sources"):
                for src in result["sources"]:
                    st.write(f"• {src}")

        if result["out_of_scope"]:
            st.info("ℹ️ This question is outside HR policy scope.", icon="ℹ️")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

# Sidebar
with st.sidebar:
    st.header("📋 HR Policy Documents")
    docs_list = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct", "Performance Review Policy",
        "Compensation & Benefits Policy", "IT & Data Security Policy",
        "POSH Policy", "Onboarding & Separation Policy", "Travel & Expense Policy"
    ]
    for doc in docs_list:
        st.write(f"📄 {doc}")
    st.divider()
    st.caption("Zyro Dynamics Pvt. Ltd. | HR Help Desk")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
