import base64
import os
import re
import shutil
import tempfile
import uuid

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

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

load_dotenv()

STAGING_ROOT = "./chroma_ui_builds"
UPLOAD_ROOT = "./temp_uploads"


def get_groq_api_key() -> str:
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secret_key = None
    if isinstance(secret_key, str) and secret_key.strip():
        return secret_key.strip()
    env_key = os.getenv("GROQ_API_KEY", "")
    return env_key.strip() if isinstance(env_key, str) else ""


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def clean_extracted_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


def _init_session_state() -> None:
    defaults = {
        "vector_store": None,
        "agent_executor": None,
        "chunks": [],
        "chroma_dir": None,
        "page_count": 0,
        "pdf_bytes": None,
        "pdf_name": None,
        "app_mode": "study",
        "messages": [],
        "chat_history": None,
        "quiz_questions": None,
        "quiz_submitted": False,
        "quiz_answers_state": {},
        "last_quiz_result": None,
        "status_message": None,
        "status_level": "info",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_document_artifacts(pdf_bytes: bytes, api_key: str):
    def _build(staging_dir: str):
        os.makedirs(staging_dir, exist_ok=True)
        os.makedirs(UPLOAD_ROOT, exist_ok=True)

        tmp_pdf_path = os.path.join(UPLOAD_ROOT, f"{uuid.uuid4().hex}.pdf")
        try:
            with open(tmp_pdf_path, "wb") as f:
                f.write(pdf_bytes)

            loader = PyPDFLoader(tmp_pdf_path)
            pages = loader.load()
            page_count = len(pages)

            for page in pages:
                page.page_content = clean_extracted_text(page.page_content)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(pages)
            cleaned_chunks = [doc.page_content for doc in split_docs if doc.page_content.strip()]

            if not cleaned_chunks:
                return {
                    "pdf_bytes": pdf_bytes,
                    "page_count": page_count,
                    "chunks": [],
                    "vector_store": None,
                    "agent_executor": None,
                }

            embeddings = get_embeddings_model()
            vector_store = Chroma.from_texts(
                texts=cleaned_chunks,
                embedding=embeddings,
                persist_directory=staging_dir,
            )

            llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0.2,
                api_key=api_key,
            )

            def pdf_search_function(query: str) -> str:
                docs = vector_store.similarity_search(query, k=3)
                return "\n\n".join(d.page_content for d in docs)

            pdf_tool = Tool(
                name="PDF_Document_Search",
                func=pdf_search_function,
                description=PDF_TOOL_DESCRIPTION,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", STUDY_SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(llm, [pdf_tool], prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=[pdf_tool],
                verbose=False,
                handle_parsing_errors=True,
            )

            return {
                "pdf_bytes": pdf_bytes,
                "page_count": page_count,
                "chunks": cleaned_chunks,
                "vector_store": vector_store,
                "agent_executor": agent_executor,
            }
        finally:
            if os.path.exists(tmp_pdf_path):
                try:
                    os.remove(tmp_pdf_path)
                except OSError:
                    pass

    return _build


def _remove_tree(path: str) -> None:
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def _render_pdf_preview(pdf_bytes: bytes) -> None:
    if not pdf_bytes:
        st.info("Upload and initialize a PDF to view it here.")
        return
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
        f'width="100%" height="520" type="application/pdf"></iframe>'
    )
    st.markdown(pdf_display, unsafe_allow_html=True)


def _render_study_lounge() -> None:
    st.subheader("Study Lounge")

    if st.session_state.agent_executor is None:
        st.info("Upload a PDF on the left and click **Initialize System** to start studying.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a question about your PDF...")
    if user_question:
        vector_store = st.session_state.vector_store

        def _search(q: str, k: int = 4):
            if vector_store is None:
                return []
            return vector_store.similarity_search(q, k=k)

        context = select_study_context(user_question, st.session_state.chunks, _search)
        grounded_input = ground_study_question(user_question, context)

        def _invoke(model_text: str, history_messages):
            return st.session_state.agent_executor.invoke(
                {
                    "input": model_text,
                    "chat_history": history_messages,
                }
            )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                error, answer = record_chat_turn(
                    st.session_state,
                    user_question,
                    _invoke,
                    model_input=grounded_input,
                )
            if error:
                st.error(error)
            else:
                st.markdown(answer)


def _generate_quiz() -> None:
    api_key = get_groq_api_key()
    if not api_key:
        st.error("Missing GROQ_API_KEY. Add it to Streamlit Secrets or your environment.")
        return

    planned = plan_quiz(st.session_state.chunks, st.session_state.page_count)
    if planned is None:
        st.warning("Not enough extractable text in the document to generate a quiz.")
        return

    context, num_questions = planned
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        api_key=api_key,
    )
    prompt = build_quiz_prompt(context, num_questions)

    with st.spinner(f"Generating {num_questions} quiz questions..."):
        try:
            response = llm.invoke(prompt)
            raw_content = getattr(response, "content", response)
            if isinstance(raw_content, list):
                raw_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                )
        except Exception as exc:
            st.error(f"Could not generate quiz: {exc}")
            return

    questions, error_kind = interpret_quiz_model_output(raw_content)
    if error_kind or not questions:
        st.error("The model returned an invalid quiz format. Please try generating again.")
        return

    st.session_state.quiz_questions = questions
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers_state = {}
    st.session_state.last_quiz_result = None


def _render_quiz_panel() -> None:
    st.subheader("Quiz Evaluation Panel")

    if not st.session_state.chunks:
        st.info("Initialize a PDF first to generate a quiz.")
        return

    col_gen, col_reset = st.columns([2, 1])
    with col_gen:
        if st.button("Generate Quiz", type="primary"):
            _generate_quiz()
    with col_reset:
        if st.session_state.quiz_questions and st.button("Reset Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers_state = {}
            st.session_state.last_quiz_result = None
            st.rerun()

    questions = st.session_state.quiz_questions
    if not questions:
        return

    with st.form("quiz_form"):
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx + 1}. {q['question']}**")
            current_choice = st.session_state.quiz_answers_state.get(idx)
            default_index = q["options"].index(current_choice) if current_choice in q["options"] else 0
            selected = st.radio(
                label=f"Options for Q{idx + 1}",
                options=q["options"],
                index=default_index,
                key=f"quiz_radio_{idx}",
                label_visibility="collapsed",
            )
            st.session_state.quiz_answers_state[idx] = selected

        submitted = st.form_submit_button("Submit Quiz")
        if submitted:
            result = grade_quiz(questions, st.session_state.quiz_answers_state)
            st.session_state.last_quiz_result = result
            st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted and st.session_state.last_quiz_result:
        result = st.session_state.last_quiz_result
        st.success(
            f"Score: **{result['correct']} / {result['total']}** ({result['percent']}%)"
        )
        for idx, row in enumerate(result["rows"]):
            icon = "✅" if row["correct"] else "❌"
            st.markdown(f"**{icon} Q{idx + 1}. {row['question']}**")
            st.write(f"- Your answer: {row['chosen']}")
            st.write(f"- Correct answer: {row['key']}")
            st.write(f"- Explanation: {row['explanation']}")


def main() -> None:
    st.set_page_config(page_title="AI Study Assistant", layout="wide")
    _init_session_state()

    st.title("AI Study Assistant")

    if st.session_state.status_message:
        if st.session_state.status_level == "error":
            st.error(st.session_state.status_message)
        elif st.session_state.status_level == "warning":
            st.warning(st.session_state.status_message)
        else:
            st.success(st.session_state.status_message)

    with st.sidebar:
        st.header("Document Setup")
        api_key = get_groq_api_key()
        if not api_key:
            st.warning("`GROQ_API_KEY` not detected yet. Add it in Streamlit Secrets or `.env`.")

        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if st.button("Initialize System", type="primary"):
            if uploaded_file is None:
                st.session_state.status_message = "Please choose a PDF file first."
                st.session_state.status_level = "warning"
            else:
                pdf_bytes = uploaded_file.getvalue()
                os.makedirs(STAGING_ROOT, exist_ok=True)
                staging_dir = os.path.join(STAGING_ROOT, f"build_{uuid.uuid4().hex}")
                build_fn = _build_document_artifacts(pdf_bytes, api_key)
                err = run_document_initialization(
                    st.session_state,
                    api_key=api_key,
                    staging_dir=staging_dir,
                    build=build_fn,
                    remove_tree=_remove_tree,
                )
                if err:
                    st.session_state.status_message = err
                    st.session_state.status_level = "error"
                else:
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.status_message = (
                        f"Initialized **{uploaded_file.name}** ({st.session_state.page_count} pages)."
                    )
                    st.session_state.status_level = "success"
                st.rerun()

        st.divider()
        st.header("Mode")
        mode_choice = st.radio(
            "Choose workspace mode",
            options=["study", "quiz"],
            format_func=lambda m: "Study Lounge" if m == "study" else "Quiz Evaluation Panel",
            index=0 if st.session_state.app_mode == "study" else 1,
        )
        st.session_state.app_mode = mode_choice

    left_col, right_col = st.columns([1, 1.3])
    with left_col:
        st.subheader("PDF Viewer")
        if st.session_state.pdf_name:
            st.caption(f"File: {st.session_state.pdf_name} | Pages: {st.session_state.page_count}")
        _render_pdf_preview(st.session_state.pdf_bytes)

    with right_col:
        if st.session_state.app_mode == "study":
            _render_study_lounge()
        else:
            _render_quiz_panel()


if __name__ == "__main__":
    main()