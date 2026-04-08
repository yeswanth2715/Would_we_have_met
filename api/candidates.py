from fastapi import APIRouter, HTTPException, Query

from api.database import get_checkin, list_open_checkins
from api.models import (
    CandidateMatch,
    ConnectionIntent,
    Mood,
    MoodCheckIn,
    SocialBattery,
    TimeHorizon,
    shared_interests_for,
)

router = APIRouter()

COMPATIBLE_MOOD_PAIRS = {
    frozenset((Mood.curious, Mood.open)),
    frozenset((Mood.energized, Mood.curious)),
    frozenset((Mood.reflective, Mood.quiet)),
}


def calculate_compatibility(user1: MoodCheckIn, user2: MoodCheckIn) -> int:
    score = 0
    shared_interests = shared_interests_for(user1, user2)

    if user1.intent == user2.intent:
        score += 30

    if frozenset((user1.mood, user2.mood)) in COMPATIBLE_MOOD_PAIRS:
        score += 25

    if user1.time_horizon == user2.time_horizon:
        score += 20

    if user1.social_battery == user2.social_battery:
        score += 15

    if user1.online_only == user2.online_only:
        score += 10

    score += min(len(shared_interests) * 8, 24)

    return score


def calculate_serendipity(user1: MoodCheckIn, user2: MoodCheckIn) -> int:
    bonus = 0

    if user1.intent == user2.intent and user1.mood != user2.mood:
        bonus += 10

    if user1.online_only != user2.online_only:
        bonus += 10

    if user1.time_horizon != user2.time_horizon:
        bonus += 5

    return min(bonus, 25)


def _load_checkin(user_id: str) -> MoodCheckIn:
    payload = get_checkin(user_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No check-in found")
    return MoodCheckIn(**payload)


@router.get("/candidates/{user_id}", response_model=list[CandidateMatch])
def get_candidates(user_id: str, limit: int = Query(default=5, ge=1, le=20)) -> list[CandidateMatch]:
    current_user = _load_checkin(user_id)
    candidates: list[CandidateMatch] = []

    for payload in list_open_checkins(exclude_user_id=user_id):
        candidate = MoodCheckIn(**payload)
        if not candidate.is_open:
            continue

        compatibility_score = calculate_compatibility(current_user, candidate)
        serendipity_bonus = calculate_serendipity(current_user, candidate)
        shared_interests = shared_interests_for(current_user, candidate)
        total_score = compatibility_score + serendipity_bonus

        candidates.append(
            CandidateMatch(
                user_id=candidate.user_id,
                display_name=candidate.display_name or candidate.user_id,
                compatibility_score=compatibility_score,
                serendipity_bonus=serendipity_bonus,
                total_score=total_score,
                mood=candidate.mood,
                intent=candidate.intent,
                time_horizon=candidate.time_horizon,
                social_battery=candidate.social_battery,
                shared_interests=shared_interests,
            )
        )

    candidates.sort(key=lambda candidate: candidate.total_score, reverse=True)
    return candidates[:limit]
