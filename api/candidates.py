from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.models import (
    ConnectionIntent,
    Mood,
    MoodCheckIn,
    SocialBattery,
    TimeHorizon,
)

router = APIRouter()

checkins_store: dict[str, dict] = {}

COMPATIBLE_MOOD_PAIRS = {
    frozenset((Mood.curious, Mood.open)),
    frozenset((Mood.energized, Mood.curious)),
    frozenset((Mood.reflective, Mood.quiet)),
}


class CandidateMatch(BaseModel):
    user_id: str
    compatibility_score: int
    serendipity_bonus: int
    total_score: int
    mood: Mood
    intent: ConnectionIntent
    time_horizon: TimeHorizon
    social_battery: SocialBattery


def set_checkins_store(store: dict[str, dict]) -> None:
    global checkins_store
    checkins_store = store


def calculate_compatibility(user1: MoodCheckIn, user2: MoodCheckIn) -> int:
    score = 0

    if user1.intent == user2.intent:
        score += 30

    if frozenset((user1.mood, user2.mood)) in COMPATIBLE_MOOD_PAIRS:
        score += 25

    if user1.time_horizon == user2.time_horizon:
        score += 20

    if user1.social_battery == user2.social_battery:
        score += 15

    if user1.is_open and user2.is_open:
        score += 10

    return score


def calculate_serendipity(user1: MoodCheckIn, user2: MoodCheckIn) -> int:
    bonus = 0

    if user1.intent == user2.intent and user1.mood != user2.mood:
        bonus += 10

    if user1.online_only != user2.online_only:
        bonus += 10

    return min(bonus, 20)


def _load_checkin(user_id: str) -> MoodCheckIn:
    payload = checkins_store.get(user_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No check-in found")
    return MoodCheckIn(**payload)


@router.get("/candidates/{user_id}", response_model=list[CandidateMatch])
def get_candidates(user_id: str, limit: int = Query(default=5, ge=1)) -> list[CandidateMatch]:
    current_user = _load_checkin(user_id)
    candidates: list[CandidateMatch] = []

    for candidate_id, payload in checkins_store.items():
        if candidate_id == user_id:
            continue

        candidate = MoodCheckIn(**payload)
        if not candidate.is_open:
            continue

        compatibility_score = calculate_compatibility(current_user, candidate)
        serendipity_bonus = calculate_serendipity(current_user, candidate)
        total_score = compatibility_score + serendipity_bonus

        candidates.append(
            CandidateMatch(
                user_id=candidate.user_id,
                compatibility_score=compatibility_score,
                serendipity_bonus=serendipity_bonus,
                total_score=total_score,
                mood=candidate.mood,
                intent=candidate.intent,
                time_horizon=candidate.time_horizon,
                social_battery=candidate.social_battery,
            )
        )

    candidates.sort(key=lambda candidate: candidate.total_score, reverse=True)
    return candidates[:limit]
