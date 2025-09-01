from typing import Dict, Any

EVENT_WEIGHTS: Dict[str, float] = {
    "eye_off_screen": 0.6,
    "tab_switch": 0.8,
    "suspicious_app": 0.9,
    "microphone_muted": 0.5,
    "network_flap": 0.3,
}

def normalize_event(event_type: str, metadata: dict) -> float:
    base = EVENT_WEIGHTS.get(event_type, 0.2)
    duration_ms = metadata.get("duration_ms", 0)
    impact = base + min(duration_ms / 5000.0, 0.4)
    return min(1.0, impact)
