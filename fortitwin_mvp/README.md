# FortiTwin MVP (Rebuilt)

This is a rebuilt FortiTwin MVP backend. It includes:
- FastAPI server with endpoints to start/continue/score an interview
- Mock emotion provider (Hume placeholder)
- Simple RAG that reads local text files
- In-memory session store and CLI runner
- Offline fallback when OpenAI API key is not provided

How to run:
1. Create and activate a virtualenv
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
2. Install requirements
   pip install -r requirements.txt
3. (Optional) Add OPENAI_API_KEY and HUME_API_KEY to a .env file
4. Ingest sample docs (optional):
   python -m app.rag --ingest sample_docs
5. Run the API:
   uvicorn app.main:app --reload --port 8000
6. Or run the CLI:
   python -m app.cli
