import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="centered")

CORPUS_PATH = "./hr_docs"
REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics' "
    "internal policy documents. Your question appears to be outside the scope of "
    "what I can help with. Please contact the HR helpdesk at "
    "hr.helpdesk@zyrodynamics.com for other queries."
)

APR_GUARANTEED_CONTEXT = """
[Guaranteed Policy Context: 05 Performance Review Policy]
ANNUAL PERFORMANCE REVIEW (APR) PROCESS — ALL 7 STAGES:
Stage 1: 360 degree feedback collected from peers and subordinates — Timeline: 1 to 20 February — Owner: HR System
Stage 2: Employee self-assessment submitted on ZyroHR portal — Timeline: 1 to 10 March — Owner: Employee
Stage 3: Manager completes assessment and submits draft rating — Timeline: 11 to 20 March — Owner: Reporting Manager
Stage 4: Calibration meeting held with all L6 and above managers — Timeline: 21 to 25 March — Owner: HR and L7+ Leaders
Stage 5: Final ratings locked and confirmed by HR — Timeline: 26 to 31 March — Owner: HR
Stage 6: One-on-one feedback conversation between employee and manager — Timeline: 1 to 10 April — Owner: Manager
Stage 7: Increment and promotion letters issued — Timeline: 15 April — Owner: HR and Finance
"""

def should_inject_apr_context(question: str) -> bool:
    q_lower = question.lower()
    strong_triggers = [
        "annual performance review", "apr timeline", "apr process",
        "review timeline", "review process", "review stages",
        "360 degree", "calibration meeting", "increment and promotion letter",
        "increment letter", "promotion letter", "self-assessment",
    ]
    return any(kw in q_lower for kw in strong_triggers)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.

Follow these rules strictly:

1. SCOPE CHECK: This question is IN SCOPE if it relates in any plausible way to
   Zyro Dynamics HR policy: leave, compensation, benefits, performance, WFH,
   onboarding, separation, conduct, IT security, POSH, travel & expense, or
   company profile/structure. Only mark OUT_OF_SCOPE if the question has ZERO
   connection to HR policy — e.g. revenue/financial performance, comparing OTHER
   companies policies (Zoho, Freshworks, Salesforce), general knowledge, or
   detailed product technical feature comparisons.
   Questions about recruitment or hiring processes are OUT_OF_SCOPE.

2. If truly out of scope, respond with EXACTLY this line and nothing else:
   OUT_OF_SCOPE: I'm sorry, I can only answer HR-related questions based on Zyro Dynamics' internal policy documents. Your question appears to be outside the scope of what I can help with. Please contact the HR helpdesk at hr.helpdesk@zyrodynamics.com for other queries.

3. If in scope, answer using ONLY the context provided. Follow ALL these rules:

   COMPANY NAME: Some documents may refer to the company as "Acrux Dynamics" due
   to template reuse. Always treat this as "Zyro Dynamics". NEVER mention or flag
   this naming difference in your answer under any circumstances.

   EXACT FACTS: Reproduce specific numbers, dates, percentages, and grade criteria
   EXACTLY as written — never paraphrase or summarise.

   APR TIMELINE: When asked about the Annual Performance Review (APR) timeline or
   process, list ALL 7 stages from the Guaranteed Policy Context with exact dates
   and owners. Never list only a subset of stages.

   PIP: When asked about PIP, state: triggered when an employee receives a rating
   of 1 or 2 in two consecutive review cycles. Duration: 60 to 90 days as
   determined by the reporting manager and HR Business Partner.

   WFH ELIGIBILITY: State ALL criteria and exclusions exactly:
   - Eligible: permanent employees at grade L3 and above, minimum 6 months service,
     Meets Expectations rating or above, no active PIP or disciplinary proceedings.
   - NOT eligible: employees on probation, employees at grades L1 and L2, employees
     deployed at client sites (unless approved in writing by HR Director).
   - Types: Hybrid WFH (L3+, max 3 days/week), Full Remote (L5+, max 5 days,
     case-by-case), Ad-hoc WFH (L3+, max 2 days, unplanned), Emergency WFH
     (all employees, as directed by HR).

   ESOP: State exactly: offered to employees at grade L5 and above, with a 4-year
   vesting schedule on a 1-year cliff basis. Do NOT add any comment about the
   number of stock options not being specified.

   TIMELINES & PROCESSES: List EVERY step with exact date range and owner.

   BENEFITS: State coverage amounts, who is covered, premium arrangements exactly.

   NO HEDGING: Do not add disclaimers about what the context does not contain
   unless the information is genuinely absent. Never add phrases like "not specified
   in the provided context" when the question only asks about facts that ARE present.

   SOURCE LINE: End every in-scope answer with: Source: <document name(s)>

Context from HR Policy Documents:
{context}

Employee Question: {question}

Answer:
""")


@st.cache_resource(show_spinner="Loading HR policies and building knowledge base...")
def load_rag_pipeline():
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    os.environ["GROQ_API_KEY"] = groq_api_key
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.4}
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
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
    docs = retriever.invoke(question)
    context = format_docs(docs)
    if should_inject_apr_context(question):
        context = APR_GUARANTEED_CONTEXT + "\n\n---\n\n" + context
    answer = parser.invoke(llm.invoke(RAG_PROMPT.invoke({"context": context, "question": question})))
    if answer.strip().startswith("OUT_OF_SCOPE:"):
        return {"answer": REFUSAL_MESSAGE, "sources": [], "out_of_scope": True}
    sources = list(set([
        d.metadata.get("source", "").split("/")[-1].replace(".pdf", "").replace("_", " ")
        for d in docs
    ]))
    return {"answer": answer, "sources": sources, "out_of_scope": False}


st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Powered by RAG + Groq (Llama 3.3 70B) | Ask any HR policy question")
st.divider()

try:
    retriever, llm = load_rag_pipeline()
    st.success("✅ HR knowledge base loaded successfully!")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm your Zyro Dynamics HR assistant. Ask me anything about our HR policies — leave, compensation, WFH, performance reviews, onboarding, and more! 👋",
        "sources": []
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.write(f"• {src}")

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)
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
        "role": "assistant", "content": result["answer"], "sources": result["sources"]
    })

with st.sidebar:
    st.header("📋 HR Policy Documents")
    for doc in ["Company Profile", "Employee Handbook", "Leave Policy",
                "Work From Home Policy", "Code of Conduct", "Performance Review Policy",
                "Compensation & Benefits Policy", "IT & Data Security Policy",
                "POSH Policy", "Onboarding & Separation Policy", "Travel & Expense Policy"]:
        st.write(f"📄 {doc}")
    st.divider()
    st.caption("Zyro Dynamics Pvt. Ltd. | HR Help Desk")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
