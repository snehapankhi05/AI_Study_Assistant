import inspect

import pytest
from langchain_community.chat_message_histories import ChatMessageHistory

from study_logic import (
    CHARS_PER_QUESTION,
    CONTEXT_CHAR_CEILING,
    MAX_QUIZ_QUESTIONS,
    grade_quiz,
    ground_study_question,
    interpret_quiz_model_output,
    plan_quiz,
    record_chat_turn,
    run_document_initialization,
    select_study_context,
    validate_quiz_payload,
)


def _session():
    history = ChatMessageHistory()
    history.add_user_message("old question")
    history.add_ai_message("old answer")
    return SimpleSession(
        pdf_bytes=b"old-pdf",
        page_count=3,
        vector_store={"id": "old-store"},
        agent_executor={"id": "old-agent"},
        chunks=["old chunk"],
        messages=[{"role": "user", "content": "old question"}],
        chat_history=history,
        quiz_questions=[{"question": "stale"}],
        quiz_submitted=True,
        quiz_answers_state={0: "stale"},
        app_mode="quiz",
        chroma_dir="live-dir",
    )


class SimpleSession:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _snapshot(session):
    return {
        "pdf_bytes": session.pdf_bytes,
        "page_count": session.page_count,
        "vector_store": session.vector_store,
        "agent_executor": session.agent_executor,
        "chunks": list(session.chunks),
        "messages": list(session.messages),
        "chat_history": session.chat_history,
        "quiz_questions": session.quiz_questions,
        "quiz_submitted": session.quiz_submitted,
        "quiz_answers_state": dict(session.quiz_answers_state),
        "app_mode": session.app_mode,
        "chroma_dir": session.chroma_dir,
    }


def test_failed_build_leaves_session_and_live_index_untouched():
    session = _session()
    before = _snapshot(session)
    removed = []

    def build(_staging):
        raise RuntimeError("loader failed")

    error = run_document_initialization(
        session,
        api_key="test-key",
        staging_dir="stage-dir",
        build=build,
        remove_tree=removed.append,
    )

    assert error
    assert "loader failed" in error
    assert _snapshot(session) == before
    assert removed == ["stage-dir"]


def test_missing_api_key_does_not_build_or_delete():
    session = _session()
    before = _snapshot(session)
    removed = []
    called = []

    def build(_staging):
        called.append(True)
        return {}

    error = run_document_initialization(
        session,
        api_key="  ",
        staging_dir="stage-dir",
        build=build,
        remove_tree=removed.append,
    )

    assert "GROQ_API_KEY" in error
    assert called == []
    assert removed == []
    assert _snapshot(session) == before


def test_empty_chunks_do_not_delete_the_live_index():
    session = _session()
    before = _snapshot(session)
    removed = []

    def build(_staging):
        return {"chunks": ["  "], "pdf_bytes": b"new", "page_count": 1, "vector_store": object(), "agent_executor": object()}

    error = run_document_initialization(
        session,
        api_key="test-key",
        staging_dir="stage-dir",
        build=build,
        remove_tree=removed.append,
    )

    assert error == "No extractable text found in this PDF."
    assert removed == ["stage-dir"]
    assert _snapshot(session) == before


def test_success_commits_every_field_and_replaces_chat_history():
    session = _session()
    old_history = session.chat_history
    removed = []
    new_store = {"id": "new-store"}
    new_agent = {"id": "new-agent"}

    def build(staging):
        assert staging == "stage-dir"
        return {
            "pdf_bytes": b"new-pdf",
            "page_count": 2,
            "vector_store": new_store,
            "agent_executor": new_agent,
            "chunks": ["Chapter one", "Chapter two"],
        }

    error = run_document_initialization(
        session,
        api_key="test-key",
        staging_dir="stage-dir",
        build=build,
        remove_tree=removed.append,
    )

    assert error is None
    assert session.pdf_bytes == b"new-pdf"
    assert session.page_count == 2
    assert session.vector_store is new_store
    assert session.agent_executor is new_agent
    assert session.chunks == ["Chapter one", "Chapter two"]
    assert session.chroma_dir == "stage-dir"
    assert session.messages == []
    assert session.chat_history is not old_history
    assert session.chat_history.messages == []
    assert session.quiz_questions is None
    assert session.quiz_submitted is False
    assert session.quiz_answers_state == {}
    assert session.app_mode == "study"
    assert removed == ["live-dir"]


def test_chat_failure_does_not_record_either_side():
    session = _session()
    before_messages = list(session.messages)
    before_count = len(session.chat_history.messages)

    def invoke(_prompt, _history):
        raise RuntimeError("quota")

    error, output = record_chat_turn(session, "What is chapter two?", invoke)

    assert output is None
    assert "quota" in error
    assert session.messages == before_messages
    assert len(session.chat_history.messages) == before_count


def test_chat_success_records_user_then_assistant():
    session = _session()
    session.messages = []
    session.chat_history = ChatMessageHistory()

    def invoke(prompt, history):
        assert prompt == "Where is it?"
        assert history == []
        return {"output": "On page 2."}

    error, output = record_chat_turn(session, "Where is it?", invoke)

    assert error is None
    assert output == "On page 2."
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.messages[1]["content"] == "On page 2."
    assert len(session.chat_history.messages) == 2


def test_chat_turn_sends_grounded_text_and_stores_the_original_question():
    session = _session()
    session.messages = []
    session.chat_history = ChatMessageHistory()
    grounded = ground_study_question("Explain the projects", "Project Atlas ships in May.")

    def invoke(prompt, history):
        assert prompt == grounded
        assert history == []
        return {"output": "Project Atlas ships in May."}

    error, output = record_chat_turn(
        session,
        "Explain the projects",
        invoke,
        model_input=grounded,
    )

    assert error is None
    assert output == "Project Atlas ships in May."
    assert session.messages[0]["content"] == "Explain the projects"
    assert session.chat_history.messages[0].content == "Explain the projects"


def test_broad_project_question_uses_saved_chunks():
    calls = []

    def search(query, k):
        calls.append((query, k))
        return ["unused search hit"]

    context = select_study_context(
        "Explain the projects in this PDF",
        ["ALPHA project one", "middle section", "OMEGA project two"],
        search,
    )

    assert calls == []
    assert "ALPHA project one" in context
    assert "OMEGA project two" in context
    assert len(context) <= CONTEXT_CHAR_CEILING


def test_specific_question_searches_four_chunks():
    def search(query, k):
        assert query == "When was the bridge finished?"
        assert k == 4
        return ["The bridge finished in 1998."]

    context = select_study_context("When was the bridge finished?", ["unrelated chunk"], search)

    assert "1998" in context
    assert "unrelated chunk" not in context


def test_empty_search_falls_back_to_saved_chunks():
    context = select_study_context("When was the bridge finished?", ["ALPHA notes", "OMEGA notes"], lambda *_args: [])

    assert "ALPHA notes" in context
    assert "OMEGA notes" in context


def test_grounded_question_says_the_pdf_is_already_loaded():
    text = ground_study_question("Explain the projects", "Project Atlas ships in May.")

    assert "already loaded" in text.lower()
    assert "do not ask the student to upload" in text.lower()
    assert "Project Atlas ships in May." in text
    assert "Explain the projects" in text


def test_chat_without_an_agent_records_nothing():
    session = _session()
    session.agent_executor = None
    before = list(session.messages)

    error, output = record_chat_turn(session, "Hello", lambda *_args: {"output": "nope"})

    assert output is None
    assert "not ready" in error
    assert session.messages == before


def _question(**overrides):
    item = {
        "question": "What is the capital?",
        "options": ["A) Paris", "B) Rome", "C) Oslo", "D) Lima"],
        "answer": "A) Paris",
        "explanation": "The notes name Paris.",
    }
    item.update(overrides)
    return item


def test_validate_quiz_payload_stores_the_option_index():
    validated = validate_quiz_payload([_question(), _question(answer="B) Rome", question="Second?")])

    assert [item["answer_index"] for item in validated] == [0, 1]
    assert validated[0]["answer"] == "A) Paris"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "not a list"},
        [],
        [_question(question=" ")],
        [_question(options=["only one"])],
        [_question(options=["A) Paris", "A) Paris"])],
        [_question(answer="A")],
        [_question(explanation="")],
        ["not a question"],
        [_question(), _question(answer="missing")],
    ],
)
def test_validate_quiz_payload_rejects_the_whole_list(payload):
    assert validate_quiz_payload(payload) is None


def test_interpret_fenced_json_and_rejects_unreadable_text():
    raw = """```json
[{"question": "Q?", "options": ["A) 1", "B) 2"], "answer": "A) 1", "explanation": "Because the text says so."}]
```"""
    questions, error = interpret_quiz_model_output(raw)
    assert error is None
    assert questions[0]["answer_index"] == 0

    questions, error = interpret_quiz_model_output("not json")
    assert questions is None
    assert error == "read"

    questions, error = interpret_quiz_model_output([{"question": "Q?"}])
    assert questions is None
    assert error == "read"


def test_plan_quiz_empty_chunks_returns_none():
    assert plan_quiz([], 4) is None
    assert plan_quiz(["   ", ""], 4) is None


def test_plan_quiz_keeps_the_first_and_last_chunk_and_takes_no_query():
    assert "query" not in inspect.signature(plan_quiz).parameters
    context, count = plan_quiz(["ALPHA unique", "middle text " * 40, "OMEGA unique"], 3)
    assert "ALPHA unique" in context
    assert "OMEGA unique" in context
    assert count >= 1


def test_plan_quiz_does_not_inflate_a_short_sample_to_the_page_count():
    sample = "w" * (CHARS_PER_QUESTION + 50)
    context, count = plan_quiz([sample], 40)
    assert count <= MAX_QUIZ_QUESTIONS
    assert count == len(context) // CHARS_PER_QUESTION
    assert count < 5


def test_plan_quiz_long_one_page_sample_asks_at_least_five():
    sample = "s" * (CHARS_PER_QUESTION * 5)
    _context, count = plan_quiz([sample], 1)
    assert count >= 5
    assert count <= MAX_QUIZ_QUESTIONS


def test_plan_quiz_caps_a_huge_sample_at_eight():
    _context, count = plan_quiz(["s" * 20000], 80)
    assert count == MAX_QUIZ_QUESTIONS


def test_grade_matching_option_index():
    questions = validate_quiz_payload([_question(), _question(question="Other?", answer="C) Oslo")])
    report = grade_quiz(questions, {0: "A) Paris", 1: "B) Rome"})

    assert report["correct"] == 1
    assert report["total"] == 2
    assert report["percent"] == 50
    assert report["rows"][0]["correct"] is True
    assert report["rows"][1]["correct"] is False
    assert report["rows"][1]["key"] == "C) Oslo"


def test_grade_unanswered_is_wrong_and_does_not_raise():
    questions = validate_quiz_payload([_question()])
    report = grade_quiz(questions, {0: None})

    assert report["correct"] == 0
    assert report["rows"][0]["chosen"] is None
    assert report["rows"][0]["key"] == "A) Paris"


def test_grade_rejects_empty_and_unindexed_questions_before_division():
    with pytest.raises(ValueError, match="empty"):
        grade_quiz([], {})
    with pytest.raises(ValueError, match="answer index"):
        grade_quiz([{"options": ["A) Paris", "B) Rome"], "answer": "A) Paris"}], {0: "A) Paris"})
