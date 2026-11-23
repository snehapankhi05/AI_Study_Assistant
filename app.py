from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 💥 FIX: Import the agent engines from langchain_classic
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

import os
import re
from dotenv import load_dotenv

print("--- Starting Agentic RAG Chatbot with Memory ---")
load_dotenv()

pdf_filename = "sample.pdf"

def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

# --- STEP 7A: Standard Ingestion (Same as before) ---
loader = PyPDFLoader(pdf_filename)
pages = loader.load()
for page in pages:
    page.page_content = clean_extracted_text(page.page_content)
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
# --- STEP 7B: Define the PDF Search Tool ---
def pdf_search_function(query: str) -> str:
    """This function is executed whenever the Agent decides to read the PDF."""
    docs = vector_store.similarity_search(query, k=2)
    return "\n\n".join([d.page_content for d in docs])

# We wrap the function in a LangChain Tool object with a very strict description
pdf_tool = Tool(
    name="PDF_Document_Search",
    func=pdf_search_function,
    description="USE THIS TOOL ONLY when the user asks a question about facts, details, history, or specific data contained inside their uploaded PDF notes/resume."
)
tools = [pdf_tool]

# --- STEP 7C: The Smart Agent Prompt with Memory Placeholders ---
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a brilliant AI study assistant sitting next to a student reading their PDF document.
    
    RULES OF ENGAGEMENT:
    1. If the student asks about facts or details inside their notes, you MUST use the 'PDF_Document_Search' tool to get the factual context.
    2. If the student asks a follow-up, abstract question, or uses terms NOT found in the document, DO NOT call the tool. Instead, use your vast internal general knowledge to explain it to them as an expert tutor, keeping it relevant to their learning context."""),
    # This placeholder tells LangChain where to inject the live rolling chat history
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    # This placeholder is required for agents to write down their internal thought processes
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Assemble the Agent Engine
agent = create_tool_calling_agent(llm, tools, agent_prompt)
# verbose=True lets us watch the AI decide whether to use the tool or its brain!
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- STEP 7D: Initialize Memory Trackers ---
chat_history_db = ChatMessageHistory()

# --- STEP 7E: The Continuous Chat Loop ---
print("\n🤖 Assistant: System is active! Ask me anything about the document or general topics. Type 'exit' to stop.\n")

while True:
    user_input = input("🙋 User: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
        
    if user_input.strip() == "":
        continue
        
    # Execute the agent, passing both the live question AND our historical memory list
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history_db.messages
    })
    
    # Update our historical memory tracker with this turn's data
    chat_history_db.add_user_message(user_input)
    chat_history_db.add_ai_message(response["output"])
    
    print(f"\n🤖 Assistant: {response['output']}\n" + "-"*40)