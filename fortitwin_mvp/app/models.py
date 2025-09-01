from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

class StartInterviewRequest(BaseModel):
    candidate_id: str
    job_title: str
    company: str
    personality: str = "Default Manager"
    rag_query: Optional[str] = None

class StartInterviewResponse(BaseModel):
    session_id: str
    first_question: str
    mode: str = Field(description="llm|offline")

class NextQuestionRequest(BaseModel):
    session_id: str
    candidate_answer: str

class NextQuestionResponse(BaseModel):
    session_id: str
    question: str
    hints: Optional[Dict[str, Any]] = None

class ScoreRequest(BaseModel):
    session_id: str

class ScoreResponse(BaseModel):
    session_id: str
    scores: Dict[str, Any]

class SecurityEvent(BaseModel):
    session_id: str
    event_type: str
    metadata: Dict[str, Any] = {}

class EmotionSignal(BaseModel):
    session_id: str
    signals: Dict[str, float]

class Session(BaseModel):
    id: str
    candidate_id: str
    job_title: str
    company: str
    personality: str
    started_at: datetime
    transcript: List[Dict[str, str]] = []
    security_events: List[SecurityEvent] = []
    emotion_context: Dict[str, float] = {}
    rag_context: str = ""
    mode: str = "offline"

class SessionStore:
    def __init__(self):
        self._store: Dict[str, Session] = {}

    def create(self, candidate_id: str, job_title: str, company: str, personality: str, rag_context: str, mode: str) -> Session:
        sid = str(uuid.uuid4())
        sess = Session(
            id=sid,
            candidate_id=candidate_id,
            job_title=job_title,
            company=company,
            personality=personality,
            started_at=datetime.utcnow(),
            rag_context=rag_context,
            mode=mode,
        )
        self._store[sid] = sess
        return sess

    def get(self, session_id: str) -> Session:
        if session_id not in self._store:
            raise KeyError("session not found")
        return self._store[session_id]

    def update(self, session: Session):
        self._store[session.id] = session

SESSION_STORE = SessionStore()
