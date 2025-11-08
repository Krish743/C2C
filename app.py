from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import os
import hashlib
import logging
from dotenv import load_dotenv

# ---- LangChain / Chroma ----
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader as DocxLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---- LLM + Prompts ----
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---- Memory + Chain ----
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# ---------- Setup ----------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("campus-sahayak")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN or ""

BASE_DIR = Path(__file__).parent.resolve()
CHROMA_DIR = str(BASE_DIR / "chroma_db")

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------- Embeddings (with CUDA if available) ----------
_device = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        _device = "cuda"
except Exception:
    _device = "cpu"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": _device}
)


# ---------- Chroma (Vector DB) ----------
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding,
)
collection = vector_store._collection  # used only to replace existing chunks for a file


# ---------- LLM + Prompts + Memory ----------
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, streaming=True)

q_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the user's latest question, rephrase the latest question as a standalone question that can be answered independently."),
    ("user", "Chat history:\n{chat_history}\n\nUser question:\n{question}")
])

qa_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an institute FAQ bot. Answer the question based on the context below. "
               "If the answer is not present in context, say 'I don't know, contact support@campussahayak.edu'."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# ✅ Compatible ConversationalRetrievalChain for LangChain 1.0.3
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    condense_question_prompt=q_prompt_template,
    combine_docs_chain_kwargs={"prompt": qa_prompt_template}
)


# ---------- Helper Function (for ingestion) ----------
def ingest_file_to_chroma(file_path: Path, department: str, category: str, filename: str):
    try:
        log.info(f"[INGEST] Start: {filename} ({department}/{category})")
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = DocxLoader(str(file_path))
        elif ext == ".doc":
            log.error("`.doc` not supported by Docx2txt. Skipping.")
            return
        else:
            loader = TextLoader(str(file_path))

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        ids, texts, metadatas = [], [], []
        for i, ch in enumerate(chunks):
            meta = ch.metadata or {}
            meta.update({
                "department": department,
                "category": category,
                "filename": filename,
                "source": str(file_path),
            })
            metadatas.append(meta)
            texts.append(ch.page_content)
            ids.append(hashlib.sha1(f"{filename}|{department}|{category}|{i}".encode()).hexdigest())

        # Replace existing vectors for this file to avoid duplicates
        collection.delete(where={
            "filename": filename,
            "department": department,
            "category": category
        })

        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        vector_store.persist()

        log.info(f"[INGEST] Done: {filename} | chunks={len(ids)}")
    except Exception as e:
        log.exception(f"[INGEST] Failed for {filename}: {e}")


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    department: str = Form(...),
    category: str = Form(...)
):
    """
    Save uploaded file and ingest it directly into the Chroma DB.
    """
    save_dir = Path(f"uploads/{department}/{category}")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file.filename

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and process
    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
    # Convert to safe path and explicitly set encoding with fallback
        safe_path = Path(file_path).as_posix()
        loader = TextLoader(safe_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    vectordb.add_documents(chunks)
    vectordb.persist()

    return {"message": f"✅ {file.filename} uploaded and ingested successfully for {department} ({category})"}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message")

    response = qa_chain({"question": user_input})
    return JSONResponse({"reply": response["answer"]})
