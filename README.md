# 🤖 Intelligent PDF Study Assistant & Multi-Agentic Quiz Engine

An advanced, enterprise-grade **Agentic RAG (Retrieval-Augmented Generation)** conversational platform designed to transform regular, static PDF documents into deeply interactive, high-fidelity educational workspaces. 

Built natively in Python using the `langchain-classic` runtime orchestrator, Groq Cloud ultra-low-latency API, HuggingFace sentence transformers, and a persistent `ChromaDB` vector engine, this application allows users to review complex technical documents side-by-side with an active AI consulting agent, shifting cleanly on-demand into an isolated, dynamically scaled knowledge evaluation panel.

---

## 🌟 Key Features

* 📖 **Side-by-Side Unified Workspace:** Features direct, web-native rendering of your uploaded PDF file on the left pane via Base64 HTML `<iframe>` streaming, synchronized alongside a multi-turn chat console on the right pane.
* 🧠 **Autonomous Agentic Routing:** Leverages LangChain's tool-calling paradigms (`AgentExecutor`) to dynamically process inputs. The agent intelligently chooses whether to call a localized vector database tool for facts or rely on internal parametric weights for conceptual, abstract explanations.
* 🏆 **Dynamic Evaluation Suite:** Ingests the document layout to scale test lengths relative to your notes (minimum 5 questions). Radio button indexes initialize strictly as `None` to ensure an unbiased examination environment without pre-selected values.
* 📊 **Granular Diagnostic Performance Reporting:** Submitting the evaluation outputs immediate scorecard metric tallies, color-coded grading, and custom LLM-generated rationale blocks detailing *why* individual options are factually right or wrong based on the source text.
* 🛡️ **Token-Constrained Memory Engineering:** Bypasses standard hardware and pricing limits (Groq's 12,000 Tokens Per Minute limit) by slicing extracted semantic context frames down to a strict 3,500-character ceiling—completely eliminating Payload Too Large (`413`) API status crashes.
* 🧹 **Self-Cleaning Storage Lifecycle:** Uses localized directory checking routines to clear past vector index databases (`./chroma_ui_db`) automatically the moment a fresh file is ingested to prevent old context data-bleeding.

---

## 🛠️ Complete Project Stack

| Component Block | Selected Technology | Target Use Case |
| :--- | :--- | :--- |
| **Interface UI** | Streamlit | Handles split-screen workspace states, multi-choice form bindings, and session caching. |
| **Framework Orchestration** | LangChain (`classic` / `core`) | Coordinates autonomous tool routing loops and stores active conversational text history. |
| **Inference Runtime Engine** | Groq Cloud API | Drives rapid, low-latency completions using the `llama-3.3-70b-versatile` model. |
| **Vector Index DB** | ChromaDB | Manages fast mathematical indexing and similarity distance calculations. |
| **Text Embedding Weights** | HuggingFace (`all-MiniLM-L6-v2`) | Compresses string chunks into multi-dimensional localized dense vectors. |
| **Data Parsing Utilities** | PyPDF, OS, Re, JSON | Powers binary stream reading, layout normalization regex, and JSON structures. |

---

## 🚀 Local Installation & Setup

Follow these quick setup steps to get the environment running on your machine:

### 1. Clone the Workspace & Enter Folder
```bash
git clone [https://github.com/your-username/Pdf-AI-Assistant.git](https://github.com/your-username/Pdf-AI-Assistant.git)
cd Pdf-AI-Assistant