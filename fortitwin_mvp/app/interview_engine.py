import os
import logging
import json
from typing import Dict, Any, Optional, List

import google.generativeai as genai
from .emotion_engine import MockEmotionProvider, HumeEmotionProvider
from .security_events import normalize_event

logger = logging.getLogger("fortitwin")

# -------------------------------------------------------------------
# Personality presets
# -------------------------------------------------------------------
PERSONALITY_PRESETS = {
    "Default Manager": dict(tone="professional", difficulty="medium"),
    "Startup CTO": dict(tone="direct", difficulty="high"),
    "FAANG Manager": dict(tone="structured", difficulty="high"),
    "Finance Recruiter": dict(tone="formal", difficulty="medium"),
}

# -------------------------------------------------------------------
# Interview Engine (Gemini + Offline fallback)
# -------------------------------------------------------------------
class InterviewEngine:
    def __init__(self):
        self.emotion = HumeEmotionProvider()
        self.fallback_emotion = MockEmotionProvider()

        key = os.getenv("GEMINI_API_KEY")
        if key:
            try:
                genai.configure(api_key=key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")
                self.mode = "gemini"
            except Exception as e:
                logger.error(f"Failed to init Gemini client: {e}")
                self.client = None
                self.mode = "offline"
        else:
            logger.warning("GEMINI_API_KEY not set, using offline mode.")
            self.client = None
            self.mode = "offline"

        logger.info(f"InterviewEngine initialized in {self.mode} mode.")

    # ------------------- Core LLM Call -------------------
    def _llm_call(self, system: str, user: str) -> str:
        if self.mode == "gemini" and self.client:
            try:
                prompt = f"{system}\n\n{user}"
                resp = self.client.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                logger.error(f"Gemini call failed: {e}")
                return ""
        return ""  # offline fallback trigger

    # ------------------- Question Generation -------------------
    def first_question(self, job_title: str, company: str, personality: str, rag_context: str) -> str:
        return self._generate_question(
            job_title, company, personality, rag_context,
            prev_answer=None,
            emotion_ctx=self._emotion_ctx("seed"),
            security_hint=None,
        )

    def next_question(
        self,
        job_title: str,
        company: str,
        personality: str,
        rag_context: str,
        prev_answer: str,
        emotion_ctx: Dict[str, float],
        security_hint: Optional[str]
    ) -> str:
        return self._generate_question(job_title, company, personality, rag_context, prev_answer, emotion_ctx, security_hint)

    def _emotion_ctx(self, session_id: str) -> Dict[str, float]:
        sig = self.emotion.get_signals(session_id)
        if not sig:
            sig = self.fallback_emotion.get_signals(session_id)
        return sig

    def _persona(self, personality: str) -> Dict[str, str]:
        return PERSONALITY_PRESETS.get(personality, PERSONALITY_PRESETS["Default Manager"])

    def _offline_question(
        self,
        job_title: str,
        persona: Dict[str, str],
        rag_context: str,
        prev_answer: Optional[str],
        emotion_ctx: Dict[str, float],
        security_hint: Optional[str],
    ) -> str:
        base = f"As a {persona['tone']} interviewer for a {job_title}, "
        if security_hint:
            base += f"I noticed a potential distraction ({security_hint}). Please stay focused. "
        if emotion_ctx.get("nervous", 0) > 0.5:
            base += "I'll keep it supportive. "
        if prev_answer:
            return base + "Can you dive deeper into your last answer and describe a concrete example with outcomes?"
        if rag_context:
            return base + "What experience do you have that directly matches this role? Refer to relevant projects."
        return base + "Tell me about a challenging project you led end-to-end."

    def _generate_question(
        self,
        job_title: str,
        company: str,
        personality: str,
        rag_context: str,
        prev_answer: Optional[str],
        emotion_ctx: Dict[str, float],
        security_hint: Optional[str],
    ) -> str:
        persona = self._persona(personality)

        system = (
            "You are an empathetic, professional AI interviewer. "
            "Ask one concise question at a time. Adjust tone/difficulty based on signals. "
            "If 'security_hint' is present, remind the candidate to focus in a respectful way."
        )

        user_parts: List[str] = [
            f"Job Title: {job_title} at {company}.",
            f"Personality: tone={persona['tone']}, difficulty={persona['difficulty']}.",
        ]
        if rag_context:
            user_parts.append(f"Company/Role Context:\n{rag_context[:2000]}")
        if prev_answer:
            user_parts.append(f"Candidate previous answer:\n{prev_answer}")
        if emotion_ctx:
            user_parts.append(f"Emotion signals: {emotion_ctx}")
        if security_hint:
            user_parts.append(f"Security hint: {security_hint}")
        user_parts.append("Now produce ONE next interview question, natural and specific.")
        prompt = "\n\n".join(user_parts)

        out = self._llm_call(system, prompt)
        if out:
            return out

        return self._offline_question(job_title, persona, rag_context, prev_answer, emotion_ctx, security_hint)

    # ------------------- Scoring -------------------
    def score(self, transcript: List[Dict[str, str]], job_title: str, company: str) -> Dict[str, Any]:
        system = "You are a fair, unbiased evaluator. Return strict JSON only."
        user = f"""Evaluate this interview for a {job_title} at {company}.
Transcript:
{transcript}

Return JSON with fields: Role Fit (0-10), Culture Fit (0-10), Honesty (0-10), Communication (0-10), Notes (string)."""

        raw = self._llm_call(system, user)
        if raw:
            try:
                return json.loads(raw)
            except Exception as e:
                logger.warning(f"Score JSON parse failed: {e}")

        # Offline fallback
        return {
            "Role Fit": 7,
            "Culture Fit": 7,
            "Honesty": 7,
            "Communication": 7,
            "Notes": "Baseline offline scoring. Provide GEMINI_API_KEY for smarter evaluation."
        }

    # ------------------- Security Hints -------------------
    @staticmethod
    def security_hint_from_event(event_type: str, metadata: dict) -> str:
        impact = normalize_event(event_type, metadata)
        if impact > 0.75:
            return f"{event_type} (high impact)"
        if impact > 0.45:
            return f"{event_type} (moderate impact)"
        return f"{event_type} (low impact)"
