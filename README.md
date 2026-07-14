# Agreement Reader (RAG)

CLI tool that answers questions about a legal agreement PDF using RAG (Gemini + FAISS).

## Stack
- `langchain` + `langchain-google-genai` (Gemini 1.5 Flash, embedding-001)
- FAISS vector store
- PyPDFLoader for ingestion

## Setup
```bash
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv
```
Add to `.env`:
```
GOOGLE_API_KEY=your_key_here
```
> Remove the hardcoded API key in `embeddings/embedder.py` and load it via `os.getenv("GOOGLE_API_KEY")` instead. Rotate the exposed key.

## Usage
Set `file_path` in `main.py` to your PDF, then:
```bash
python main.py
```
Ask questions in the prompt; type `exit` to quit.

## Structure
```
main.py                    # entry point
loaders/pdf_loader.py      # PDF -> chunked documents
embeddings/embedder.py     # embeddings + FAISS vectorstore
retriever/rag_pipeline.py  # retrieval + Gemini QA chain
data/                      # source PDF
```
