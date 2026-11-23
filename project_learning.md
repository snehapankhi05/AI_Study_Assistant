# Project learning notes

This file explains the PDF AI Study Assistant as it exists in the workspace: what the product is, which libraries it uses, how a session runs, and what was changed after the first version. The short setup guide is [README.md](README.md).

## What the project is

This is a single-user study tool. A person uploads one PDF in the browser, reads it beside a chat, and can switch into a quiz generated from that same file.

It is a retrieval-augmented app. The PDF is not pasted into the model in full. The app extracts the text, splits it, embeds the pieces, and sends the model a small retrieved sample plus the student’s question. The quiz uses a sample of the same saved chunks.

The interface is Streamlit. There is one study agent, not a team of agents. The agent has one tool, `PDF_Document_Search`. The chat also retrieves text before the agent runs, because leaving retrieval optional made the model ask the user to upload a PDF that was already on screen.

`app.py` is an earlier command-line chatbot. It still expects a file named `sample.pdf` and writes `./chroma_db`. The browser app does not import it. The app you run is `ui_app.py`.

## Files

| File | Role |
| :--- | :--- |
| `ui_app.py` | Streamlit page: upload, PDF iframe, chat, quiz form, Groq and Chroma setup |
| `study_logic.py` | Testable functions: session commit, chat recording, context choice, quiz plan, JSON check, grading |
| `tests/test_study_logic.py` | Unit tests for those functions. No network and no Chroma |
| `app.py` | Old terminal loop. Kept in the folder, not used by the UI |
| `.env` | Local Groq key. Git ignores it |
| `.gitignore` | Ignores `.env`, `chroma_ui_builds/`, `chroma_db/`, `temp_uploads/`, and virtualenvs |

There is no `requirements.txt`. The import list in the README is the set of packages the UI needs.

## Libraries and models

- **Streamlit** holds the page, the sidebar, `st.session_state`, the chat box, and the quiz form.
- **python-dotenv** loads `.env` when the script starts. `load_dotenv()` does not replace a `GROQ_API_KEY` that is already set in the process.
- **PyPDFLoader** (`pypdf` underneath) reads a PDF from a real file path. The upload is written to `temp_uploads/<random>.pdf`, loaded, then deleted.
- **RecursiveCharacterTextSplitter** cuts the cleaned text into chunks of 1,000 characters with 200 characters of overlap.
- **HuggingFaceEmbeddings** with `all-MiniLM-L6-v2` turns chunks into vectors on the local machine. The model downloads on first use.
- **Chroma** (`langchain_community.vectorstores.Chroma`) stores those vectors. Each successful initialize creates a new folder under `chroma_ui_builds/<random>/`. The collection name Chroma uses by default is `langchain`.
- **LangChain classic** builds the study agent: `create_tool_calling_agent` and `AgentExecutor`, with a `ChatPromptTemplate` that includes `chat_history` and `agent_scratchpad`.
- **ChatMessageHistory** is the memory object sent back to the agent. The visible bubbles are a separate list, `session.messages`.
- **ChatGroq** calls Groq. Both the study agent and the quiz use `openai/gpt-oss-20b` at temperature `0.2`. An earlier study setting, `llama-3.3-70b-versatile`, returned HTTP 404 on the key used with this project, so the study agent was switched to the model the quiz had already called successfully.

Groq only checks that the key is present when `ChatGroq` is constructed. A bad key still gets through initialization and fails on the first real request with HTTP 401.

## How text is cleaned

`clean_extracted_text` in `ui_app.py` does two things before chunking:

1. Collapse every whitespace run into a single space.
2. Delete every character outside the ASCII range `\x00-\x7F`.

Accents, many equations, and non-English letters never reach the index. The iframe still shows the original PDF bytes, so the page can display characters the model cannot retrieve. Empty text after this step, which is what a scan usually produces, raises “No extractable text found in this PDF.” and does not replace the previous session.

## Session state

Streamlit reruns the whole script on each click. Values that must survive live on `st.session_state`:

| Name | Meaning |
| :--- | :--- |
| `pdf_bytes` | Original file, used only for the iframe |
| `page_count` | Number of PDF pages from PyPDF, including sparse pages |
| `chunks` | Cleaned chunk strings saved at initialize time |
| `vector_store` | Live Chroma client |
| `chroma_dir` | Folder for that client |
| `agent_executor` | Study agent built for this document |
| `messages` | Bubbles shown in the chat |
| `chat_history` | `ChatMessageHistory` sent to the agent |
| `app_mode` | `"study"` or `"quiz"` |
| `quiz_questions` | Validated question list, or empty |
| `quiz_submitted` | Whether the scorecard is showing |
| `quiz_answers_state` | Radio values captured on submit |

## Initialize, and why it commits once

`build_document` does the slow work into local variables and a new directory. It must not write the live session. `run_document_initialization` in `study_logic.py` is what publishes the result.

Order:

1. Refuse immediately if `GROQ_API_KEY` is missing or blank. Nothing is built.
2. Call `build(staging_dir)`. That writes the temp PDF, parses, chunks, embeds, opens Chroma, and constructs the agent.
3. If that raises, delete only the staging directory and return the error. The previous PDF, index, agent, transcript, and memory stay as they were.
4. If the chunks are empty or the build dict is incomplete, do the same.
5. On success, assign the new store, agent, chunks, page count, PDF bytes, and a brand-new empty `ChatMessageHistory` together. Quiz state is cleared and the mode returns to study. Then the previous Chroma directory is deleted on a best-effort basis.

The old design deleted `./chroma_ui_db` before the new index existed and wrote `pdf_bytes` before parsing finished. A failure in the middle showed the new PDF against the old index, and Windows often locked the SQLite file. The staging directory exists so the live index is still readable until the new one is ready. If the old folder cannot be deleted, it is leftover disk. The session pointer is what the agent searches.

The search tool closes over `st.session_state.vector_store` and reads it at call time, with `k=3`. The store and the executor are replaced in the same commit so a new question does not pair an old conversation with a new document.

## Chat

The chat box calls `select_study_context`, then `ground_study_question`, then `record_chat_turn`.

Context choice:

- A broad question (summarize, overview, “explain the projects,” “what is this about,” and the other phrases in `_BROAD_PHRASES`) uses the saved chunks. Those chunks are spread through the document, with the first and last pieces kept, and cut at 3,500 characters. This is the same spread the quiz uses.
- A specific question calls `similarity_search(question, k=4)` on the live store.
- An empty result or a search error falls back to the saved-chunk sample.

`ground_study_question` wraps that text with a fixed instruction: the PDF is already loaded, answer from the context, do not ask the student to upload or paste the file, and say what is missing if the context does not contain the answer. That string is the `input` the agent sees.

`record_chat_turn` stores the student’s original sentence in `messages` and in `chat_history`. It does not store the retrieved block. The next turn retrieves again instead of growing the history by another 3,500 characters. If the agent raises, neither transcript is updated.

The system prompt (`STUDY_SYSTEM_PROMPT`) repeats the same rule and allows `PDF_Document_Search` when the model wants another passage. The tool description tells it the PDF is already there.

`AgentExecutor` is created with `verbose=True`, so tool traces print in the terminal that launched Streamlit. Parsing errors and tool errors propagate. The UI catches them and shows “The assistant could not answer.”

## Quiz

**Launch Evaluation Quiz** calls `generate_dynamic_quiz`.

1. `plan_quiz` samples the saved chunks, same 3,500-character ceiling. Page count is ignored on purpose. A 40-page file with a short sample must not become a 40-question exam.
2. The exam length is one question per 400 characters of that sample, capped at 8. Under 400 characters, the quiz is refused and the model is not called.
3. `build_quiz_prompt` asks Groq for a raw JSON list. Each item needs `question`, `options`, `answer`, and `explanation`. The answer must be copied from that item’s options.
4. `interpret_quiz_model_output` strips a ```json fence if the model added one, parses JSON, and runs `validate_quiz_payload`.
5. Validation rejects the whole list if any item is incomplete, if options are not distinct strings, or if `answer` is not exactly one of the options. A valid item gets `answer_index`.
6. Only a non-empty validated list switches `app_mode` to `"quiz"`. A Groq error and a format error use different messages. The study page stays up.

The form uses `st.radio(..., index=None)`, so nothing is pre-selected. On submit, the chosen strings are stored and `grade_quiz` compares each choice with `options[answer_index]`. A blank answer is wrong. The correct option text is what the scorecard shows. At 70 percent or more, Streamlit shows balloons. The explanation is the one string generated with the question. It is not a second pass over the option the student picked.

## What changed from the first version

The first UI was one script, `ui_app.py`, plus the README and `app.py`. Behavior that looked right in the README did not match the code. The work after that was:

1. **Atomic initialize and chat memory.** A new PDF no longer publishes `pdf_bytes` or `page_count` before the index and agent exist. Chat memory is replaced only in that success commit, so the previous document’s conversation is not sent with the new file. Failures leave the previous session alone.
2. **Chat errors.** A failed Groq call does not leave a user bubble without an assistant reply.
3. **Quiz schema.** Malformed model JSON cannot open the quiz page. The quiz temperature was lowered from 0.7 to 0.2 so the answer string stays closer to the option text.
4. **Quiz evidence and size.** The quiz stopped using `similarity_search("")` and `max(5, page_count)`. It samples the real chunks and caps the exam at 8.
5. **Grading.** Scores use `answer_index` after the schema check, instead of hoping the model’s `answer` string matches the radio value.
6. **Model id.** The study agent moved from `llama-3.3-70b-versatile` to `openai/gpt-oss-20b` after Groq returned 404 for the Llama id on this key.
7. **Grounded chat.** The agent used to see only the question. It often replied by asking for the PDF. Each chat turn now attaches retrieved document text first. Broad questions use the chunk sample. Specific questions use vector search.

`study_logic.py` was added so those rules can be tested without booting Streamlit or calling Groq. `tests/test_study_logic.py` holds those tests.

## Design choices that are easy to misread

- **Two histories.** `messages` is the UI. `chat_history` is the model memory. They are updated together only after a successful answer, and both are cleared together when a new document commits.
- **The iframe is not the index.** The left pane is the original bytes. The model sees cleaned, chunked, retrieved text.
- **The agent is optional for the first retrieval.** The tool can fetch another passage, but the first context is attached in code. That was required because `gpt-oss-20b` would otherwise answer without calling the tool.
- **3,500 characters is a product limit, not a Groq guarantee.** It keeps one quiz or one broad answer inside a small prompt. It does not prove the request will stay under Groq’s tokens-per-minute quota, and it does not cover a long textbook.
- **Deleting the old index is cleanup.** Correctness comes from pointing the session at the new directory. A locked old folder can remain under `chroma_ui_builds/`.
- **One process, one active document.** Two browser tabs are two sessions that can both write under `chroma_ui_builds/`. They do not share a session, but they share the disk.

## How to run and how to test

```bash
pip install streamlit python-dotenv pypdf chromadb langchain-classic langchain-community langchain-core langchain-groq langchain-huggingface langchain-text-splitters sentence-transformers
```

`.env`:

```
GROQ_API_KEY=gsk_your_key
```

```bash
streamlit run ui_app.py
```

```bash
python -m pytest tests/test_study_logic.py -q -p no:cacheprovider
```

After any change to the agent prompt or the model name, restart Streamlit and click **Initialize System** again. Those objects are created during initialize and kept on the session. A code edit does not rebuild an agent that is already in memory.

## Known gaps

- Non-ASCII text is stripped.
- A long document is sampled. Eight quiz questions and a 3,500-character chat sample do not cover every page.
- There is no login, no file-size cap, and no per-user database.
- `app.py` still has the older prompt, `k=2`, and `./chroma_db`.
- The score explanation does not discuss each wrong option separately.
- Packages are not pinned in a requirements file.
