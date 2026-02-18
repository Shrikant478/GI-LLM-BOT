import os
import streamlit as st
from dotenv import load_dotenv

from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Disable LangSmith tracing (prevents 401 warnings)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)
os.environ.pop("LANGCHAIN_PROJECT", None)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")
st.title("🤖 Groq Chatbot (LangChain + Streamlit)")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing in .env")
    st.stop()

# -------- Fetch LIVE Groq models (no hardcoding) --------
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    model_ids = [m.id for m in groq_client.models.list().data]
except Exception as e:
    st.error(f"Groq connection failed: {e}")
    st.stop()

if not model_ids:
    st.error("No models returned for this Groq key. Regenerate the key and try again.")
    st.stop()

# -------- Session state --------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Proper auto-selection: remember last chosen model, else pick first available
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_ids[0]

# If previously selected model disappeared, fallback safely
if st.session_state.selected_model not in model_ids:
    st.session_state.selected_model = model_ids[0]

# -------- Sidebar --------
with st.sidebar:
    st.header("Settings")

    selected_model = st.selectbox(
        "Choose model",
        model_ids,
        index=model_ids.index(st.session_state.selected_model),
    )
    st.session_state.selected_model = selected_model

    MEMORY_LIMIT = st.slider("Memory window (last N messages)", 2, 20, 8, step=2)

    st.caption(f"Using: **{selected_model}**")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# -------- Build LLM + Chain --------
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use conversation history to answer contextually."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# -------- Display chat history --------
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# ✅ Search bar at bottom (always visible)
user_input = st.chat_input("Type your message...")

if user_input:
    # Store + show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Window memory optimization ----
    recent = st.session_state.messages[-MEMORY_LIMIT:]

    history_msgs = []
    for m in recent:
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        else:
            history_msgs.append(AIMessage(content=m["content"]))

    response = ""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({
                    "history": history_msgs,
                    "question": user_input
                })
            except Exception as e:
                response = f"Error: {e}"
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
