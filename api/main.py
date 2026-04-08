from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from api.candidates import router as candidates_router
from api.database import get_checkin, get_db_path, init_db, save_checkin
from api.mediators import router as mediators_router
from api.models import MoodCheckIn, normalize_interests

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Would We Have Met - Alpha API")
app.include_router(candidates_router)
app.include_router(mediators_router)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/checkin/mood")
def submit_mood_checkin(checkin: MoodCheckIn):
    """User submits their current mood, intent, and availability."""
    prepared = _prepare_checkin(checkin)
    payload = prepared.model_dump()
    payload["mood"] = prepared.mood.value
    payload["social_battery"] = prepared.social_battery.value
    payload["intent"] = prepared.intent.value
    payload["time_horizon"] = prepared.time_horizon.value
    payload["timestamp"] = prepared.timestamp.isoformat()

    save_checkin(payload)
    return {"status": "ok", "message": "Mood captured", "user_id": prepared.user_id}


@app.get("/checkin/mood/{user_id}")
def get_mood_checkin(user_id: str):
    """Retrieve the latest mood check-in for a user."""
    payload = get_checkin(user_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No check-in found")
    return payload


@app.get("/health")
def health():
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc),
        "database": str(get_db_path()),
    }


def _prepare_checkin(checkin: MoodCheckIn) -> MoodCheckIn:
    user_id = checkin.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    display_name = (checkin.display_name or user_id).strip()
    if not display_name:
        display_name = user_id

    conversation_style = (checkin.conversation_style or "").strip() or None

    return MoodCheckIn(
        user_id=user_id,
        display_name=display_name,
        mood=checkin.mood,
        social_battery=checkin.social_battery,
        intent=checkin.intent,
        time_horizon=checkin.time_horizon,
        is_open=checkin.is_open,
        online_only=checkin.online_only,
        neighborhood_radius_km=checkin.neighborhood_radius_km,
        interests=normalize_interests(checkin.interests),
        conversation_style=conversation_style,
        timestamp=checkin.timestamp,
    )
