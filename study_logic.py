"""Pure helpers for document commit, quiz planning, and grading.

The Streamlit script calls these functions. They do not import Streamlit, so
tests can exercise the failure paths without starting the app or Chroma.
"""

import gc
import json
import re
from langchain_community.chat_message_histories import ChatMessageHistory

MAX_QUIZ_QUESTIONS = 8
CONTEXT_CHAR_CEILING = 3500
# One question needs this much sampled text. Shorter samples produce a shorter
# exam. A sample under this length is refused instead of padded to five questions.
CHARS_PER_QUESTION = 400


def run_document_initialization(session, *, api_key, staging_dir, build, remove_tree):
    """Build a document off to the side, then publish it in one commit.

    `build(staging_dir)` must return the new artifacts or raise. It must not
    write the live session. On failure the previous session is left as it was,
    and only the staging directory is removed.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        return "Missing GROQ_API_KEY. Add it to the environment or a .env file and try again."

    previous_dir = getattr(session, "chroma_dir", None)
    try:
        built = build(staging_dir)
    except Exception as exc:
        _best_effort_remove(remove_tree, staging_dir)
        return f"Could not initialize the document: {exc}"

    usable = usable_chunks(built.get("chunks") if isinstance(built, dict) else None)
    if not usable:
        _best_effort_remove(remove_tree, staging_dir)
        return "No extractable text found in this PDF."

    required = ("pdf_bytes", "page_count", "vector_store", "agent_executor")
    if not isinstance(built, dict) or any(key not in built for key in required):
        _best_effort_remove(remove_tree, staging_dir)
        return "Could not initialize the document: the loader did not return a complete session."

    commit_loaded_document(
        session,
        {
            "pdf_bytes": built["pdf_bytes"],
            "page_count": built["page_count"],
            "vector_store": built["vector_store"],
            "agent_executor": built["agent_executor"],
            "chunks": usable,
            "chroma_dir": staging_dir,
        },
    )
    # Drop the previous client before deleting its directory. A locked SQLite
    # file is leftover disk, not the index the agent still searches.
    gc.collect()
    if previous_dir and _same_path(previous_dir, staging_dir) is False:
        _best_effort_remove(remove_tree, previous_dir)
    return None


def commit_loaded_document(session, built):
    """Publish a finished document. Call this only after build has succeeded."""
    session.vector_store = built["vector_store"]
    session.agent_executor = built["agent_executor"]
    session.chunks = list(built["chunks"])
    session.chroma_dir = built["chroma_dir"]
    session.page_count = built["page_count"]
    session.quiz_questions = None
    session.quiz_submitted = False
    session.quiz_answers_state = {}
    session.messages = []
    session.chat_history = ChatMessageHistory()
    session.pdf_bytes = built["pdf_bytes"]
    session.app_mode = "study"


STUDY_SYSTEM_PROMPT = (
    "You are an integrated AI tutor. The student's PDF is already loaded and indexed. "
    "Retrieved excerpts are included with each question. Answer from those excerpts. "
    "Call PDF_Document_Search when you need another passage from the same PDF. "
    "Never ask the student to upload, paste, or share the PDF. "
    "If the retrieved text does not contain the answer, say what is missing."
)

PDF_TOOL_DESCRIPTION = (
    "Search the PDF that is already loaded. Use this for projects, sections, "
    "explanations, facts, and any other detail from the student's document. "
    "Do not ask the student to provide the PDF."
)

_BROAD_PHRASES = (
    "summar",
    "overview",
    "sum up",
    "tl;dr",
    "what is this document",
    "what's this document",
    "what is this pdf",
    "what is this about",
    "what's this about",
    "explain the project",
    "explain my project",
    "explain these project",
    "explain this document",
    "explain this pdf",
    "describe the document",
    "describe this pdf",
    "list the project",
    "list my project",
    "projects in",
    "walk me through",
    "main points",
    "key points",
)


def record_chat_turn(session, user_prompt, invoke, model_input=None):
    """Append a chat turn only after the agent returns output.

    `model_input` is what the agent sees. The transcript stores `user_prompt`,
    so retrieved PDF text is not pasted back into the next turn.

    Returns `(error, output)`. On error both transcripts stay unchanged.
    """
    if getattr(session, "agent_executor", None) is None:
        return "The study agent is not ready. Initialize a PDF first.", None
    sent = user_prompt if model_input is None else model_input
    try:
        response = invoke(sent, session.chat_history.messages)
        if not isinstance(response, dict) or "output" not in response:
            raise ValueError("the agent returned no output")
        output = response["output"]
        if not isinstance(output, str):
            output = str(output)
    except Exception as exc:
        return f"The assistant could not answer: {exc}", None

    session.messages.append({"role": "user", "content": user_prompt})
    session.messages.append({"role": "assistant", "content": output})
    session.chat_history.add_user_message(user_prompt)
    session.chat_history.add_ai_message(output)
    return None, output


def is_broad_question(question):
    """A document-wide request should sample the saved chunks, not one neighborhood."""
    text = " ".join(str(question).lower().split())
    if any(phrase in text for phrase in _BROAD_PHRASES):
        return True
    asks_for_projects = "project" in text and any(
        word in text for word in ("explain", "list", "describe", "overview", "summar", "what")
    )
    return asks_for_projects


def sample_document_context(chunks):
    spread = _spread_texts(usable_chunks(chunks))
    return _fit_context(spread, CONTEXT_CHAR_CEILING)


def select_study_context(question, chunks, search):
    """Return document text for one chat turn, capped at the quiz context ceiling.

    `search(query, k)` is used for a specific question. A broad question, an
    empty search result, or a search error falls back to the saved chunks.
    """
    if is_broad_question(question) or search is None:
        return sample_document_context(chunks)
    try:
        hits = search(question, 4) or []
    except Exception:
        return sample_document_context(chunks)
    parts = []
    for hit in hits:
        if isinstance(hit, str):
            text = hit.strip()
        else:
            text = str(getattr(hit, "page_content", "")).strip()
        if text:
            parts.append(text)
    joined = "\n\n".join(parts).strip()
    if not joined:
        return sample_document_context(chunks)
    return joined[:CONTEXT_CHAR_CEILING]


def ground_study_question(question, context):
    body = (context or "").strip() or "No document text was retrieved."
    return (
        "The student's PDF is already loaded. Answer from the retrieved context below. "
        "Do not ask the student to upload, paste, or share the PDF. "
        "If the context does not contain the answer, say what is missing.\n\n"
        f"Retrieved context:\n{body}\n\n"
        f"Student question:\n{question}"
    )


def usable_chunks(chunks):
    usable = []
    for chunk in chunks or []:
        if isinstance(chunk, str):
            text = chunk.strip()
        else:
            text = str(getattr(chunk, "page_content", "")).strip()
        if text:
            usable.append(text)
    return usable


def plan_quiz(chunks, page_count):
    """Choose quiz evidence from saved chunks, in document order.

    Page count is accepted so callers can keep passing it, and it is not used
    as a minimum. A 40-page PDF with a short sample must not grow the exam.
    """
    del page_count
    spread = _spread_texts(usable_chunks(chunks))
    context = _fit_context(spread, CONTEXT_CHAR_CEILING)
    if len(context) < CHARS_PER_QUESTION:
        return None
    count = min(MAX_QUIZ_QUESTIONS, len(context) // CHARS_PER_QUESTION)
    return context, count


def build_quiz_prompt(context, num_questions):
    return f"""
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

    The answer value must be copied exactly from that question's options list.

    Context:
    {context}
    """


def strip_model_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def validate_quiz_payload(data):
    """Accept a quiz only when every item is complete. Otherwise reject it all."""
    if not isinstance(data, list) or not data:
        return None
    normalized = []
    for item in data:
        checked = _validate_question(item)
        if checked is None:
            return None
        normalized.append(checked)
    return normalized


def interpret_quiz_model_output(content):
    """Return `(questions, error_kind)`. `error_kind` is `'read'` or None."""
    if not isinstance(content, str):
        return None, "read"
    try:
        payload = json.loads(strip_model_json(content))
    except json.JSONDecodeError:
        return None, "read"
    validated = validate_quiz_payload(payload)
    if validated is None:
        return None, "read"
    return validated, None


def grade_quiz(questions, selections):
    """Grade by the option index stored during schema validation."""
    if not questions:
        raise ValueError("Cannot grade an empty quiz.")
    rows = []
    correct = 0
    for index, item in enumerate(questions):
        if not isinstance(item, dict) or "answer_index" not in item or "options" not in item:
            raise ValueError("Cannot grade a question that has no answer index.")
        options = item["options"]
        answer_index = item["answer_index"]
        if (
            not isinstance(options, list)
            or not options
            or not isinstance(answer_index, int)
            or answer_index < 0
            or answer_index >= len(options)
        ):
            raise ValueError("Cannot grade a question that has no answer index.")
        key = options[answer_index]
        chosen = selections.get(index) if isinstance(selections, dict) else None
        is_correct = chosen == key
        if is_correct:
            correct += 1
        rows.append(
            {
                "question": item.get("question", ""),
                "explanation": item.get("explanation", ""),
                "chosen": chosen,
                "key": key,
                "correct": is_correct,
            }
        )
    total = len(questions)
    return {
        "correct": correct,
        "total": total,
        "percent": int((correct / total) * 100),
        "rows": rows,
    }


def _validate_question(item):
    if not isinstance(item, dict):
        return None
    question = item.get("question")
    options = item.get("options")
    answer = item.get("answer")
    explanation = item.get("explanation")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(options, list) or len(options) < 2:
        return None
    if not all(isinstance(option, str) and option.strip() for option in options):
        return None
    if len(set(options)) != len(options):
        return None
    if not isinstance(answer, str) or answer not in options:
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    return {
        "question": question.strip(),
        "options": list(options),
        "answer": answer,
        "answer_index": options.index(answer),
        "explanation": explanation.strip(),
    }


def _spread_texts(texts):
    if len(texts) <= 3:
        return list(texts)
    slots = min(len(texts), 8)
    chosen = []
    seen = set()
    for slot in range(slots):
        index = round(slot * (len(texts) - 1) / (slots - 1))
        if index not in seen:
            seen.add(index)
            chosen.append(texts[index])
    return chosen


def _fit_context(texts, ceiling):
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0][:ceiling]
    last = texts[-1]
    reserve = min(len(last), max(1, ceiling // 4))
    body_limit = max(0, ceiling - reserve - 2)
    parts = []
    used = 0
    for piece in texts[:-1]:
        separator = 2 if parts else 0
        room = body_limit - used - separator
        if room <= 0:
            break
        take = piece[:room]
        parts.append(take)
        used += separator + len(take)
        if len(take) < len(piece):
            break
    tail = last[:reserve]
    if not parts:
        return tail[:ceiling]
    return "\n\n".join(parts + [tail])[:ceiling]


def _best_effort_remove(remove_tree, path):
    if not path:
        return
    try:
        remove_tree(path)
    except Exception:
        return


def _same_path(left, right):
    return str(left).replace("\\", "/").rstrip("/") == str(right).replace("\\", "/").rstrip("/")