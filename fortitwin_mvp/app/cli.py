from .models import SESSION_STORE
from .interview_engine import InterviewEngine
from .rag import retrieve

def run_cli():
    print("FortiTwin CLI — interactive interview (type 'quit' to exit)")
    job_title = input("Job Title: ").strip() or "Software Engineer"
    company = input("Company: ").strip() or "Acme"
    personality = input("Personality (Default Manager / Startup CTO / FAANG Manager / Finance Recruiter): ").strip() or "Default Manager"
    rag_query = input("RAG query (optional): ").strip()

    rag_ctx = retrieve(rag_query) if rag_query else ""
    engine = InterviewEngine()
    session = SESSION_STORE.create(candidate_id="cli", job_title=job_title, company=company, personality=personality, rag_context=rag_ctx, mode=engine.mode)

    q = engine.first_question(job_title, company, personality, rag_ctx)
    print(f"\nInterviewer: {q}")

    while True:
        ans = input("\nYou: ").strip()
        if ans.lower() in {"quit", "exit"}:
            break
        session.transcript.append({"role": "candidate", "text": ans})
        q = engine.next_question(job_title, company, personality, rag_ctx, ans, session.emotion_context or {"nervous":0.3,"confident":0.5,"empathetic_need":0.2}, None)
        session.transcript.append({"role": "interviewer", "text": q})
        print(f"\nInterviewer: {q}")

    scores = engine.score(session.transcript, job_title, company)
    print("\nFinal Scores:", scores)

if __name__ == '__main__':
    run_cli()
