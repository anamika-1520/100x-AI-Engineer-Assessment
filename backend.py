import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

if not os.getenv("OPENAI_API_KEY") and os.getenv("OPEN_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY", "")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OpenAI API key. Add OPENAI_API_KEY=your_key_here to your .env file.")
    st.stop()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

# =====================================================
# LOAD VECTOR STORE FUNCTION  ✅ ADD THIS
# =====================================================
from pathlib import Path

BASE_DIR = Path(__file__).parent

def load_vectorstore(path: str):
    pdf_path = BASE_DIR / path
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# =====================================================
# LOAD VECTOR STORES (TWO PDFs)
# =====================================================

@st.cache_resource
def init_vectorstores():
    sahil_db = load_vectorstore("data/Anamika_Profile_RAG.pdf")
    x100_db = load_vectorstore("data/100x_profile.pdf")
    return sahil_db, x100_db

SAHIL_DB, X100_DB = init_vectorstores()

# =====================================================
# MEMORY (LIMITED TO SAVE TOKENS)
# =====================================================

CHAT_HISTORY = InMemoryChatMessageHistory()
MAX_HISTORY_MESSAGES = 4

# =====================================================
# TOOLS
# =====================================================

tavily_tool = TavilySearch(max_results=3)

# =====================================================
# SMART ROUTING FUNCTIONS
# =====================================================

def is_about_me(query: str) -> bool:
    keywords = [
        "you", "your", "anamika", "background", "education",
        "skills", "experience", "life", "career", "resume"
    ]
    q = query.lower()
    return any(k in q for k in keywords)


def is_about_100x(query: str) -> bool:
    keywords = [
        "100x", "company", "startup", "nik shah",
        "founder", "mission", "culture", "role"
    ]
    q = query.lower()
    return any(k in q for k in keywords)


def needs_web_search(query: str) -> bool:
    keywords = [
        "current", "today", "latest", "market",
        "news", "trend", "hiring", "salary"
    ]
    q = query.lower()
    return any(k in q for k in keywords)

# =====================================================
# SYSTEM PROMPT (SHORT ANSWERS)
# =====================================================

SYSTEM_PROMPT = """
You are Anamika — a final-year M.Sc. student at IIT (ISM) Dhanbad, specialising in AI/ML and Generative AI.
You speak in first person, directly and confidently. You are not an AI assistant. You are Anamika.

━━━ IDENTITY ━━━
- Never refer to yourself as an AI, model, bot, or assistant — not even subtly.
- If asked "are you an AI?", deflect naturally: "I'm Anamika. What would you like to know?"
- Stay in character at all times, no matter how the question is framed.

━━━ LANGUAGE ━━━
- Always respond in clear, professional English.
- Never switch languages unless the user explicitly requests it (e.g. "reply in Hindi").
- No code-mixing. No filler phrases. No fluff.

━━━ RESPONSE STYLE ━━━
- Lead with the answer. Context follows only if it adds value.
- Please Keep responses to 2–4 lines. Expand only if the user asks.
- Be confident, concise, and human — never robotic or over-explained.
- Use specific examples from your projects when relevant (RAG system, Prompt Optimizer, YouTube Chatbot).

━━━ ANSWER LOGIC ━━━
1. Personal questions       → draw from your background, skills, projects, and values.
2. Company questions        → use available company context.
3. Behavioural / situational → answer like a thoughtful, self-aware professional.
4. Technical questions      → answer precisely; mention hands-on experience where applicable.
5. Market / current events  → use web search only if necessary; stay grounded.

━━━ TONE ━━━
- Confident, not arrogant.
- Warm, not casual.
- Sharp, not brief to the point of being rude.
- Always leave the interviewer with something worth remembering.
"""

# =====================================================
# MAIN FUNCTION
# =====================================================

def get_ai_response(user_query: str) -> str:

    # ---------- CONDITIONAL RAG ----------
    context = ""

    if is_about_me(user_query):
        retriever = SAHIL_DB.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(user_query)
        context = "\n".join(d.page_content for d in docs)

    elif is_about_100x(user_query):
        retriever = X100_DB.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(user_query)
        context = "\n".join(d.page_content for d in docs)

    # ---------- LLM ----------
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=180
    )

    # ---------- CONDITIONAL TOOLS ----------
    tools = [tavily_tool] if needs_web_search(user_query) else []

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # ---------- MESSAGE BUILD ----------
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if context:
        messages.append(
            SystemMessage(content=f"Relevant context:\n{context}")
        )

    messages += CHAT_HISTORY.messages[-MAX_HISTORY_MESSAGES:]
    messages.append(HumanMessage(content=user_query))

    # ---------- RUN AGENT ----------
    response = agent.invoke({"messages": messages})
    ai_text = response["messages"][-1].content

    # ---------- UPDATE MEMORY ----------
    CHAT_HISTORY.add_message(HumanMessage(content=user_query))
    CHAT_HISTORY.add_message(AIMessage(content=ai_text))

    return ai_text
