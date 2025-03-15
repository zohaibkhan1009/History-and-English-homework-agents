import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import PyPDF2
import pandas as pd
from docx import Document

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Wikipedia API for external knowledge
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Agent 1: History Expert (Indian and World History)
history_llm = ChatOpenAI(temperature=0, api_key=API_KEY, model="gpt-4o-mini")
history_tools = [wikipedia_tool]
history_agent = create_react_agent(history_llm, history_tools)

# Agent 2: English Expert (Language and Literature)
english_llm = ChatOpenAI(temperature=0, api_key=API_KEY, model="gpt-4o-mini")
english_tools = [wikipedia_tool]
english_agent = create_react_agent(english_llm, english_tools)

# Streamlit UI Setup
st.set_page_config(page_title="Chat with Expert", layout="wide")

# Sidebar Navigation
st.sidebar.subheader("Select an Expert")
agent_choice = st.sidebar.radio("Choose an Expert:", ["History Expert", "English Expert"])

def clear_chat():
    st.session_state.chat_history = []

def stop_session():
    st.stop()

st.sidebar.button("Clear Chat History", on_click=clear_chat)
st.sidebar.button("End Session", on_click=stop_session)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 Chat with Expert")
st.subheader(f"Chat with the {agent_choice}")
user_input = st.text_input("Ask a question:")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.write("Thinking...")
    
    if agent_choice == "History Expert":
        response = history_agent.invoke({"messages": [HumanMessage(content=user_input)]})
    else:
        response = english_agent.invoke({"messages": [HumanMessage(content=user_input)]})
    
    reply = response["messages"][-1].content
    st.session_state.chat_history.append(("bot", reply))

# Display Chat History
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"<div style='text-align: right; background-color: #4CAF50; color: white; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; background-color: #f1f1f1; color: black; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message}</div>", unsafe_allow_html=True)

# File Upload and Summarization Feature
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif uploaded_file.type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        if uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        text = df.to_string()
    else:
        text = "Unsupported file format."
    return text

st.sidebar.subheader("Upload a File")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOC, CSV, etc.")
if uploaded_file:
    st.sidebar.success("File uploaded successfully!")
    extracted_text = extract_text_from_file(uploaded_file)
    
    if extracted_text and extracted_text != "Unsupported file format.":
        summary_prompt = "Summarize the following text: " + extracted_text[:2000]
        if agent_choice == "History Expert":
            summary_response = history_agent.invoke({"messages": [HumanMessage(content=summary_prompt)]})
        else:
            summary_response = english_agent.invoke({"messages": [HumanMessage(content=summary_prompt)]})
        
        summary = summary_response["messages"][-1].content
        st.sidebar.subheader("Summary")
        st.sidebar.write(summary)
    else:
        st.sidebar.error("Could not extract text from the uploaded file.")

st.sidebar.subheader("Theme Selection")
dark_mode = st.sidebar.toggle("Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #1e1e1e; color: white; } .stChatContainer { background-color: #333; } .stChatBubble { background-color: #444; color: white; }</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body { background-color: white; color: black; } .stChatContainer { background-color: white; } .stChatBubble { background-color: #f1f1f1; color: black; }</style>", unsafe_allow_html=True)

st.success("Setup Complete! Run 'streamlit run app.py' to launch the chatbot.")
