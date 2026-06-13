import os
import shutil
import re
import json
import base64
import streamlit as st
from dotenv import load_dotenv

# Core Ingestion & Agent Elements
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Classic Runtime Framework for Agents
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Page Config
st.set_page_config(page_title="Study Workspace", page_icon="📖", layout="wide")
load_dotenv()

# --- 2. GLOBAL SESSION STATES ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0

# App Mode: "study" or "quiz"
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "study"

# Quiz Specific Tracking
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_answers_state" not in st.session_state:
    st.session_state.quiz_answers_state = {}

def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

# --- 3. DYNAMIC QUIZ GENERATION LOGIC ---
# --- 3. DYNAMIC QUIZ GENERATION LOGIC (OPTIMIZED FOR GROQ TPM LIMITS) ---
def generate_dynamic_quiz():
    if st.session_state.vector_store is None:
        return None
    
    # Scale question count based on PDF size (Min 5 questions)
    num_questions = max(5, st.session_state.page_count)
    
    # 💥 FIX: Lower k to 3 and cap maximum context characters to stay safely under Groq's 12K TPM limit
    docs = st.session_state.vector_store.similarity_search("", k=3)
    raw_context = "\n\n".join([d.page_content for d in docs])
    context_text = raw_context[:3500]  # Hard character ceiling (~800-900 tokens max context)
    
    quiz_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    quiz_prompt = f"""
    You are an expert academic evaluator. Based on the document context provided, create exactly {num_questions} unique multiple-choice questions.
    Vary the topics across the text.

    Return your answer STRICTLY as a raw, valid JSON list of dictionaries. Do not wrap it in markdown block quotes (do not add ```json). No conversation.

    JSON Structure:
    [
      {{
        "question": "Clear test question text?",
        "options": ["A) Choice 1", "B) Choice 2", "C) Choice 3", "D) Choice 4"],
        "answer": "A) Choice 1",
        "explanation": "Detailed explanation of why this answer is correct based on the text."
      }}
    ]

    Context:
    {context_text}
    """
    
    try:
        response = quiz_llm.invoke(quiz_prompt)
        clean_text = response.content.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(clean_text)
    except Exception as e:
        st.error(f"Failed parsing randomized quiz content. Try triggering again. Error: {e}")
        return None# --- 4. FILE INGESTION PIPELINE ---
with st.sidebar:
    st.header("📁 Document Intake")
    uploaded_file = st.file_uploader("Drop a study PDF here", type=["pdf"])
    
    if st.button("Initialize System", type="primary", disabled=not uploaded_file):
        with st.spinner("Analyzing document mapping..."):
            temp_dir = "./temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Read bytes for rendering and write copy to disk for PyPDF
            bytes_data = uploaded_file.read()
            st.session_state.pdf_bytes = bytes_data
            
            with open(temp_pdf_path, "wb") as f:
                f.write(bytes_data)
                
            loader = PyPDFLoader(temp_pdf_path)
            pages = loader.load()
            st.session_state.page_count = len(pages)
            
            for page in pages:
                page.page_content = clean_extracted_text(page.page_content)
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)
            
            if os.path.exists("./chroma_ui_db"):
                shutil.rmtree("./chroma_ui_db")
                
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_ui_db")
            
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
            
            def pdf_search_function(query: str) -> str:
                docs = st.session_state.vector_store.similarity_search(query, k=3)
                return "\n\n".join([d.page_content for d in docs])

            pdf_tool = Tool(
                name="PDF_Document_Search",
                func=pdf_search_function,
                description="Use this when verifying specific data inside the document rules."
            )
            
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an integrated AI tutor. Help the student read their document. Use PDF_Document_Search when asked explicit facts about the text."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_tool_calling_agent(llm, [pdf_tool], agent_prompt)
            st.session_state.agent_executor = AgentExecutor(agent=agent, tools=[pdf_tool], verbose=True)
            
            # Clear historical context tags
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.session_state.messages = []
            st.session_state.app_mode = "study"
            st.success("System loaded successfully!")

    # Navigation option shortcuts inside sidebar
    if st.session_state.vector_store is not None:
        st.write("---")
        st.header("⚡ Mode Selector")
        if st.session_state.app_mode == "study":
            if st.button("🏆 Launch Evaluation Quiz", use_container_width=True):
                with st.spinner("Drafting fresh randomized questions..."):
                    st.session_state.quiz_questions = generate_dynamic_quiz()
                    if st.session_state.quiz_questions:
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers_state = {}
                        st.session_state.app_mode = "quiz"
                        st.rerun()
        else:
            if st.button("⬅️ Return to Study Lounge", use_container_width=True):
                st.session_state.app_mode = "study"
                st.rerun()

# =========================================================
# --- VIEW MODE 1: SIDE-BY-SIDE STUDY LOUNGE ---
# =========================================================
if st.session_state.app_mode == "study":
    if st.session_state.pdf_bytes is not None:
        # Create dual workspace split columns
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.subheader("📖 Document Viewer")
            # Convert PDF stream to safely render in browser container
            base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" style="border:none; border-radius:8px;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
        with right_col:
            st.subheader("💬 Active Consultation Chat")
            
            # Container with set height for clear conversational flow
            chat_container = st.container(height=680)
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        
            if user_prompt := st.chat_input("Inquire about text records..."):
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(user_prompt)
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                
                if st.session_state.agent_executor is not None:
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Extracting parameters..."):
                                response = st.session_state.agent_executor.invoke({
                                    "input": user_prompt,
                                    "chat_history": st.session_state.chat_history.messages
                                })
                                output_text = response["output"]
                                st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    st.session_state.chat_history.add_user_message(user_prompt)
                    st.session_state.chat_history.add_ai_message(output_text)
    else:
        st.info("👋 Welcome! Please upload a PDF file in the sidebar menu and select 'Initialize System' to begin your study session.")

# =========================================================
# --- VIEW MODE 2: EXCLUSIVE FULL-PAGE QUIZ ---
# =========================================================
elif st.session_state.app_mode == "quiz":
    st.subheader("🏆 Comprehensive Knowledge Evaluation Panel")
    st.markdown(f"Generated **{len(st.session_state.quiz_questions)}** personalized questions tailored to your document ({st.session_state.page_count} pages loaded).")
    st.write("---")
    
    # We use a Streamlit form to make sure options are completely blank/not evaluated beforehand
    with st.form(key="comprehensive_quiz_form"):
        temp_selections = {}
        
        for i, item in enumerate(st.session_state.quiz_questions):
            st.markdown(f"#### **Q{i+1}: {item['question']}**")
            
            # Form default: Index=None forces it to display completely blank until clicked
            temp_selections[i] = st.radio(
                f"Options for question {i+1}",
                options=item["options"],
                index=None,
                key=f"prod_quiz_idx_{i}"
            )
            st.write("")
            
        submit_trigger = st.form_submit_button("Submit Assessment Solutions", type="primary")
        
    if submit_trigger:
        st.session_state.quiz_submitted = True
        st.session_state.quiz_answers_state = temp_selections

    # Handle display evaluation after processing submission click
    if st.session_state.quiz_submitted:
        st.write("---")
        st.markdown("### 📊 Graded Scorecard Evaluation")
        
        final_correct_tallies = 0
        total_quiz_count = len(st.session_state.quiz_questions)
        
        for i, item in enumerate(st.session_state.quiz_questions):
            chosen = st.session_state.quiz_answers_state.get(i)
            correct_ans = item["answer"]
            
            st.markdown(f"**Question {i+1}:** {item['question']}")
            
            if chosen == correct_ans:
                final_correct_tallies += 1
                st.success(f"🏅 **Correct!** You selected: `{chosen}`")
            else:
                st.error(f"❌ **Incorrect.** You selected: `{chosen if chosen else 'No response chosen'}`")
                st.markdown(f"* **Correct Answer Key:** `{correct_ans}`")
                
            # Direct actionable analysis block why it's correct
            st.info(f"💡 **Analysis Detail:** {item['explanation']}")
            st.write("")
            
        # Display Final Score Card Badge
        score_percent = int((final_correct_tallies / total_quiz_count) * 100)
        if score_percent >= 70:
            st.balloons()
            st.success(f"🎉 **Great Job! Final Score:** {final_correct_tallies} / {total_quiz_count} ({score_percent}%)")
        else:
            st.warning(f"📉 **Keep Practicing! Final Score:** {final_correct_tallies} / {total_quiz_count} ({score_percent}%)")
            
    if st.button("⬅️ Return to Chat Workspace", type="secondary"):
        st.session_state.app_mode = "study"
        st.rerun()