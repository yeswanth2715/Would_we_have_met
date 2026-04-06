from fastapi import FastAPI, HTTPException
from api.models import MoodCheckIn
from datetime import datetime

app = FastAPI(title="Would We Have Met - Alpha API")

# Temporary in-memory store (replace with DB later)
checkins = {}

@app.post("/checkin/mood")
def submit_mood_checkin(checkin: MoodCheckIn):
    """User submits their current mood, intent, and availability."""
    checkins[checkin.user_id] = checkin.dict()
    return {"status": "ok", "message": "Mood captured", "user_id": checkin.user_id}

@app.get("/checkin/mood/{user_id}")
def get_mood_checkin(user_id: str):
    """Retrieve the latest mood check-in for a user."""
    if user_id not in checkins:
        raise HTTPException(status_code=404, detail="No check-in found")
    return checkins[user_id]

@app.get("/health")
def health():
    return {"status": "alive", "timestamp": datetime.utcnow()}
