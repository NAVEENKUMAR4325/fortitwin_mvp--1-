import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import (
    StartInterviewRequest, StartInterviewResponse,
    NextQuestionRequest, NextQuestionResponse,
    ScoreRequest, ScoreResponse,
    SecurityEvent, EmotionSignal,
    SESSION_STORE,
)
from .interview_engine import InterviewEngine
from .rag import retrieve

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fortitwin")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUME_API_KEY = os.getenv("HUME_API_KEY")

# Debug check (⚠️ remove in production)
if GEMINI_API_KEY:
    logger.info(f"GEMINI key loaded: {GEMINI_API_KEY[:6]}...")
else:
    logger.warning("GEMINI_API_KEY not found in .env!")

if HUME_API_KEY:
    logger.info(f"HUME key loaded: {HUME_API_KEY[:6]}...")
else:
    logger.warning("HUME_API_KEY not found in .env!")

app = FastAPI(
    title="FortiTwin MVP API",
    version="0.2.0",
    description="API backend for adaptive AI-driven interviews (Gemini + Hume)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Engine auto-detects Gemini from env
ENGINE = InterviewEngine()

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    """Health check route."""
    return {"message": "FortiTwin MVP API is running 🚀 (Gemini + Hume)"}


@app.post("/interview/start", response_model=StartInterviewResponse)
def start_interview(req: StartInterviewRequest):
    """Start a new interview session."""
    logger.info(f"Starting interview for candidate={req.candidate_id}, job_title={req.job_title}")

    rag_ctx = retrieve(req.rag_query or "") if req.rag_query else ""
    sess = SESSION_STORE.create(
        candidate_id=req.candidate_id,
        job_title=req.job_title,
        company=req.company,
        personality=req.personality,
        rag_context=rag_ctx,
        mode=ENGINE.mode,
    )

    q = ENGINE.first_question(req.job_title, req.company, req.personality, rag_ctx)
    sess.transcript.append({"role": "interviewer", "text": q})
    SESSION_STORE.update(sess)

    logger.info(f"Session {sess.id} started with first question")
    return StartInterviewResponse(session_id=sess.id, first_question=q, mode=ENGINE.mode)


@app.post("/interview/next", response_model=NextQuestionResponse)
def next_question(req: NextQuestionRequest):
    """Submit candidate’s answer and get next interview question."""
    sess = SESSION_STORE.get(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Session {sess.id}: candidate answered -> {req.candidate_answer}")
    sess.transcript.append({"role": "candidate", "text": req.candidate_answer})

    # Check for last security event
    security_hint = None
    if sess.security_events:
        last = sess.security_events[-1]
        security_hint = InterviewEngine.security_hint_from_event(last.event_type, last.metadata)

    # Default emotion context if none
    emotion_ctx = sess.emotion_context or {
        "nervous": 0.3,
        "confident": 0.5,
        "empathetic_need": 0.2,
    }

    q = ENGINE.next_question(
        sess.job_title,
        sess.company,
        sess.personality,
        sess.rag_context,
        req.candidate_answer,
        emotion_ctx,
        security_hint,
    )

    sess.transcript.append({"role": "interviewer", "text": q})
    SESSION_STORE.update(sess)

    logger.info(f"Session {sess.id}: next question generated")
    return NextQuestionResponse(
        session_id=sess.id,
        question=q,
        hints={"security_hint": security_hint, "emotion_ctx": emotion_ctx},
    )


@app.post("/interview/event")
def post_event(ev: SecurityEvent):
    """Log a security-related event during the interview."""
    sess = SESSION_STORE.get(ev.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    sess.security_events.append(ev)
    SESSION_STORE.update(sess)
    logger.info(f"Session {sess.id}: security event logged -> {ev.event_type}")
    return {"ok": True}


@app.post("/interview/emotion")
def post_emotion(sig: EmotionSignal):
    """Log emotion signals for adaptive questioning."""
    sess = SESSION_STORE.get(sig.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    sess.emotion_context = sig.signals
    SESSION_STORE.update(sess)
    logger.info(f"Session {sess.id}: emotion signals updated")
    return {"ok": True}


@app.post("/interview/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Compute and return final interview scores."""
    sess = SESSION_STORE.get(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    scores = ENGINE.score(sess.transcript, sess.job_title, sess.company)
    logger.info(f"Session {sess.id}: scoring complete")
    return ScoreResponse(session_id=sess.id, scores=scores)


# -------------------------------------------------------------------
# Ping endpoints
# -------------------------------------------------------------------
@app.get("/ping-gemini")
def ping_gemini():
    """Quick test that Gemini key works."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content("Hello from FortiTwin")
        return {"ok": True, "reply": resp.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/ping-hume")
def ping_hume():
    """Quick test that Hume key works."""
    try:
        from hume.client import HumeClient
        client = HumeClient(api_key=HUME_API_KEY)

        # Fetch available tools (pager → list)
        pager = client.empathic_voice.tools.list_tools()
        tools = list(pager)   # convert pager into list

        return {"ok": True, "reply": f"Hume connected; found {len(tools)} tools"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/ping-all")
def ping_all():
    """Test both Gemini + Hume keys."""
    results = {}
    results["gemini"] = ping_gemini()
    results["hume"] = ping_hume()
    return results
