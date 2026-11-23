# PDF AI Study Assistant

A local Streamlit app that turns one text-based PDF into a study workspace. The PDF stays on the left. A chat about that PDF stays on the right. A separate quiz panel can generate and grade multiple-choice questions from the same document.

The running app is `ui_app.py`. The rules for loading, chat context, quiz size, and grading live in `study_logic.py`. A longer explanation of the design, the files, and the changes made to the project is in [project_learning.md](project_learning.md).

## What you can do

- Upload a PDF and click **Initialize System**. The file is parsed, cleaned, split, embedded, and stored in a new Chroma directory. The screen updates only after that work succeeds.
- Ask questions in the chat. The app attaches retrieved text from the loaded PDF before the model answers, so the model is not asked to request the file again.
- Launch a quiz. Question count follows the amount of sampled text, with a maximum of 8. Each question is checked against a fixed JSON shape before it is shown. Grading uses the option index, and radios start unselected.

## Stack

| Piece | What it is used for |
| :--- | :--- |
| Streamlit | Split workspace, sidebar, chat, and quiz form |
| PyPDF | Read text from the uploaded PDF |
| LangChain text splitter | 1,000-character chunks with 200-character overlap |
| HuggingFace `all-MiniLM-L6-v2` | Local embeddings |
| Chroma | Vector index for the study search tool and specific chat questions |
| LangChain `AgentExecutor` | One study agent with one search tool |
| Groq `openai/gpt-oss-20b` | Chat answers and quiz generation |
| python-dotenv | Loads `GROQ_API_KEY` from `.env` |

## Setup

Python 3.11 is what this workspace was run with. From the project folder:

```bash
pip install streamlit python-dotenv pypdf chromadb langchain-classic langchain-community langchain-core langchain-groq langchain-huggingface langchain-text-splitters sentence-transformers
```

Create a `.env` file next to `ui_app.py`. Get a key from [Groq console keys](https://console.groq.com/keys). Do not put quotes or spaces around the value:

```
GROQ_API_KEY=gsk_your_key
```

If Windows already has `GROQ_API_KEY` set, that process value wins over `.env`. Clear the old value or start the terminal without it.

Start the app:

```bash
streamlit run ui_app.py
```

Upload a text PDF, click **Initialize System**, then chat or launch the quiz. After a code change, stop Streamlit, start it again, and initialize the PDF again. The agent is built during initialization and is kept in the browser session.

## Tests

```bash
python -m pytest tests/test_study_logic.py -q -p no:cacheprovider
```

These tests cover commit and rollback, chat recording, quiz schema, context selection, and grading. They do not call Groq or Chroma.

## Limits to know

- Scanned PDFs have no text layer, so initialization stops with “No extractable text.”
- Characters outside basic ASCII are removed before chunking.
- Chat and quiz context are capped at 3,500 characters. A long PDF is sampled, not copied in full.
- `app.py` is an older command-line script. It is not the Streamlit app.
