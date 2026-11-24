# AI Study Assistant

A Streamlit app for studying one text-based PDF. Upload a file, initialize it, then ask questions or generate a multiple-choice quiz from the extracted text.

The main script is `app.py`. Shared loading, retrieval, quiz, and grading rules are in `study_logic.py`. `ui_app.py` is a second Streamlit script with a side-by-side PDF iframe and chat pane. `requirements.txt` lists the runtime packages. There is no Dockerfile and no `.streamlit` config in this repository.

Hosted app: [https://aistudyassistant-hnhfxjicjj5rrgtqfxoskr.streamlit.app/](https://aistudyassistant-hnhfxjicjj5rrgtqfxoskr.streamlit.app/)

That address is a public Streamlit Community Cloud URL. It does not contain an API key or other secret. The app sleeps after inactivity and can be woken from the page. The Groq key used by a deployment stays in Streamlit secrets on the server.

## Features

`app.py` provides:

- A sidebar upload and **Initialize System**. The PDF is parsed only after you click that button. The session updates only if initialization succeeds.
- Study chat. Retrieved text from the loaded PDF is attached to the question before the model answers.
- A quiz mode. **Generate Quiz** builds questions from a sample of the document. Radios start with nothing selected. **Submit Quiz** scores the attempt and shows the correct option and the generated explanation.

`ui_app.py` provides the same initialize, chat, and quiz behavior through an older page layout: the original PDF is shown in an iframe beside the chat.

Both scripts use one study agent with a single search tool, `PDF_Document_Search`. Quiz length depends on how much text was sampled, with a maximum of 8 questions.

## Stack

| Piece | Use |
| :--- | :--- |
| Streamlit | Interface and session state |
| PyPDF | Text extraction |
| LangChain text splitter | 1,000-character chunks, 200-character overlap |
| HuggingFace `all-MiniLM-L6-v2` | Local embeddings. `app.py` caches the model with `st.cache_resource` |
| Chroma | Vector index, one new directory per successful initialize under `chroma_ui_builds/` |
| LangChain `AgentExecutor` | Study agent and tool call |
| Groq `openai/gpt-oss-20b` | Chat and quiz generation, temperature 0.2 |
| python-dotenv | Local `GROQ_API_KEY` from `.env` |

`app.py` reads `GROQ_API_KEY` from Streamlit secrets first, then from the environment. `ui_app.py` reads the environment only.

## Local setup

Python 3.11 was used in this workspace. From the project folder:

```bash
pip install -r requirements.txt
```

Create a `.env` file next to the scripts. Get a key from the [Groq console](https://console.groq.com/keys). Do not add quotes or spaces around `=`:

```
GROQ_API_KEY=gsk_your_key
```

`load_dotenv()` does not override a `GROQ_API_KEY` that is already set in the process. On Streamlit Community Cloud, set the same name in the app’s secrets instead of committing `.env`. `.env` is listed in `.gitignore`.

Run the main app:

```bash
streamlit run app.py
```

The alternate layout is:

```bash
streamlit run ui_app.py
```

Upload a text PDF and click **Initialize System**. Restart Streamlit and initialize again after changing the agent, model, or prompt. Those objects are built during initialization and kept in the session.

## Tests

`pytest` is not in `requirements.txt`. Install it separately, then:

```bash
python -m pytest tests/test_study_logic.py -q -p no:cacheprovider
```

The tests cover commit and rollback, chat recording, context selection, quiz validation, and grading. They do not call Groq or Chroma.

## Limits

- A scanned PDF with no text layer fails initialization with “No extractable text found in this PDF.”
- Characters outside basic ASCII are removed before chunking. The viewer, where one is shown, still uses the original file bytes.
- Chat and quiz context are capped at 3,500 characters. A long PDF is sampled, not sent in full.
- Question count is one question per 400 characters of that sample, and never more than 8. A sample shorter than 400 characters does not call the model.
- This is a single-user local or single-deployment app. Sessions do not share chat memory, but index files are written on the machine running the app.
