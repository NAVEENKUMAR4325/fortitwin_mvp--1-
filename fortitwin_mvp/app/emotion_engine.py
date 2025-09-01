import os, random
from typing import Dict, Optional

class EmotionProvider:
    def get_signals(self, session_id: str) -> Dict[str, float]:
        raise NotImplementedError

class MockEmotionProvider(EmotionProvider):
    def get_signals(self, session_id: str) -> Dict[str, float]:
        rnd = random.Random(hash(session_id) & 0xffffffff)
        nervous = rnd.uniform(0.2, 0.6)
        confident = 1.0 - nervous - rnd.uniform(0.0, 0.2)
        confident = max(0.0, min(1.0, confident))
        empathetic_need = rnd.uniform(0.1, 0.5)
        return {
            "nervous": round(nervous, 2),
            "confident": round(confident, 2),
            "empathetic_need": round(empathetic_need, 2),
        }

class HumeEmotionProvider(EmotionProvider):
    def __init__(self):
        self.api_key: Optional[str] = os.getenv("HUME_API_KEY")

    def get_signals(self, session_id: str) -> Dict[str, float]:
        if not self.api_key:
            return MockEmotionProvider().get_signals(session_id)
        # Real Hume integration would go here; for now return a neutral set
        return {"nervous": 0.3, "confident": 0.5, "empathetic_need": 0.2}
