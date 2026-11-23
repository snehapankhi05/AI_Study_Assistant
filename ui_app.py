import os
import uuid
import shutil
import re
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

from study_logic import (
    PDF_TOOL_DESCRIPTION,
    STUDY_SYSTEM_PROMPT,
    build_quiz_prompt,
    grade_quiz,
    ground_study_question,
    interpret_quiz_model_output,
    plan_quiz,
    record_chat_turn,
    run_document_initialization,
    select_study_context,
)

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
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = None
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

def remove_index_dir(path):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

def build_document(pdf_bytes, staging_dir):
    """Load and index a PDF into `staging_dir`. Do not touch the live session."""
    os.makedirs(staging_dir, exist_ok=True)
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_pdf_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.pdf")
    try:
        with open(temp_pdf_path, "wb") as handle:
            handle.write(pdf_bytes)
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        for page in pages:
            page.page_content = clean_extracted_text(page.page_content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        if not chunks:
            raise ValueError("No extractable text found in this PDF.")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(
            chunks, embeddings, persist_directory=staging_dir
        )
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

        def pdf_search_function(query: str) -> str:
            docs = st.session_state.vector_store.similarity_search(query, k=3)
            return "\n\n".join([d.page_content for d in docs])

        pdf_tool = Tool(
            name="PDF_Document_Search",
            func=pdf_search_function,
            description=PDF_TOOL_DESCRIPTION,
        )
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", STUDY_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, [pdf_tool], agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[pdf_tool], verbose=True)
        return {
            "pdf_bytes": pdf_bytes,
            "page_count": len(pages),
            "vector_store": vector_store,
            "agent_executor": agent_executor,
            "chunks": [chunk.page_content for chunk in chunks],
        }
    finally:
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass

# --- 3. DYNAMIC QUIZ GENERATION LOGIC ---
def generate_dynamic_quiz():
    if st.session_state.vector_store is None:
        return None

    planned = plan_quiz(st.session_state.chunks, st.session_state.page_count)
    if planned is None:
        st.error("This document does not have enough extractable text to build a quiz.")
        return None
    context_text, num_questions = planned

    try:
        quiz_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
        response = quiz_llm.invoke(build_quiz_prompt(context_text, num_questions))
    except Exception as exc:
        st.error(f"The quiz request failed: {exc}")
        return None

    questions, error_kind = interpret_quiz_model_output(getattr(response, "content", None))
    if error_kind == "read" or not questions:
        st.error("The quiz could not be read: the response did not match the expected question format.")
        return None
    return questions

# --- 4. FILE INGESTION PIPELINE ---
with st.sidebar:
    st.header("📁 Document Intake")
    uploaded_file = st.file_uploader("Drop a study PDF here", type=["pdf"])

    if st.button("Initialize System", type="primary", disabled=not uploaded_file):
        with st.spinner("Analyzing document mapping..."):
            pdf_bytes = uploaded_file.getvalue()
            staging_dir = os.path.join("chroma_ui_builds", uuid.uuid4().hex)
            error = run_document_initialization(
                st.session_state,
                api_key=os.getenv("GROQ_API_KEY"),
                staging_dir=staging_dir,
                build=lambda path: build_document(pdf_bytes, path),
                remove_tree=remove_index_dir,
            )
            if error:
                st.error(error)
            else:
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
                with st.spinner("Extracting parameters..."):
                    context = select_study_context(
                        user_prompt,
                        st.session_state.chunks,
                        None
                        if st.session_state.vector_store is None
                        else lambda query, k: st.session_state.vector_store.similarity_search(query, k=k),
                    )
                    error, output_text = record_chat_turn(
                        st.session_state,
                        user_prompt,
                        lambda prompt, history: st.session_state.agent_executor.invoke(
                            {"input": prompt, "chat_history": history}
                        ),
                        model_input=ground_study_question(user_prompt, context),
                    )
                if error:
                    st.error(error)
                else:
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(user_prompt)
                        with st.chat_message("assistant"):
                            st.markdown(output_text)
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
        try:
            report = grade_quiz(
                st.session_state.quiz_questions,
                st.session_state.quiz_answers_state,
            )
        except ValueError as exc:
            st.error(f"This quiz cannot be graded: {exc}")
        else:
            for index, row in enumerate(report["rows"], start=1):
                st.markdown(f"**Question {index}:** {row['question']}")
                if row["correct"]:
                    st.success(f"🏅 **Correct!** You selected: `{row['chosen']}`")
                else:
                    chosen = row["chosen"] if row["chosen"] else "No response chosen"
                    st.error(f"❌ **Incorrect.** You selected: `{chosen}`")
                    st.markdown(f"* **Correct Answer Key:** `{row['key']}`")
                st.info(f"💡 **Analysis Detail:** {row['explanation']}")
                st.write("")

            score_percent = report["percent"]
            final_correct_tallies = report["correct"]
            total_quiz_count = report["total"]
            if score_percent >= 70:
                st.balloons()
                st.success(f"🎉 **Great Job! Final Score:** {final_correct_tallies} / {total_quiz_count} ({score_percent}%)")
            else:
                st.warning(f"📉 **Keep Practicing! Final Score:** {final_correct_tallies} / {total_quiz_count} ({score_percent}%)")

    if st.button("⬅️ Return to Chat Workspace", type="secondary"):
        st.session_state.app_mode = "study"
        st.rerun()
