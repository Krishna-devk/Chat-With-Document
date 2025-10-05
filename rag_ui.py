# app.py
import streamlit as st
import asyncio
import nest_asyncio
from rag_logic import RAGPipeline
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import tempfile
import base64

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()

st.set_page_config(page_title="GramVani Chat", page_icon="🤖", layout="wide")

st.sidebar.header("📂 Upload Document (Optional)")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

output_language = st.sidebar.selectbox(
    "🌐 Choose Output Language",
    ["English", "Hindi", "Tamil", "Telugu", "Marathi", "Gujarati", "Bengali"]
)

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "qa_ready" not in st.session_state:
    st.session_state.qa_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag = st.session_state.rag

if uploaded_file and not st.session_state.qa_ready:
    with st.spinner("📄 Processing your document..."):
        try:
            docs = rag.load_file(uploaded_file)
            retriever = rag.build_retriever(docs)
            rag.create_qa_chain(retriever)
            st.session_state.qa_ready = True
            st.success("✅ File processed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

st.subheader("🎙️ Ask a Question")
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Type your question here (any language):")

with col2:
    voice_query = None
    if st.button("🎤 Record"):
        voice_query = speech_to_text(
            language='en',
            use_container_width=True,
            just_once=True,
            key='STT'
        )
        if voice_query:
            st.success(f"🎤 You said: {voice_query}")
            query = voice_query  

if query:
    with st.spinner("💭 Generating answer..."):
        try:
            answer = rag.ask(query, output_language)
            st.session_state.chat_history.append({"user": query, "bot": answer})
        except Exception as e:
            st.error(f"⚠️ Error generating answer: {str(e)}")

st.subheader("💬 Conversation")
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**GramVani:** {chat['bot']}")
    st.markdown("---")

if st.session_state.chat_history:
    last_answer = st.session_state.chat_history[-1]["bot"]
    lang_code = "en" if output_language.lower() == "english" else "hi"
    tts = gTTS(last_answer, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        audio_file = open(tmp_file.name, "rb")
        audio_bytes = audio_file.read()
        b64 = base64.b64encode(audio_bytes).decode()
        st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")

if st.session_state.qa_ready:
    st.info("📘 Document mode active — answers are based on uploaded document.")
else:
    st.info("💬 Chat mode active — general AI answers.")

if st.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
