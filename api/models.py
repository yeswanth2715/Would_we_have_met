from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Mood(str, Enum):
    quiet = "quiet"
    curious = "curious"
    open = "open"
    cautious = "cautious"
    energized = "energized"
    reflective = "reflective"


class SocialBattery(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ConnectionIntent(str, Enum):
    friendship = "friendship"
    dating = "dating"
    activity_partner = "activity_partner"
    conversation = "conversation"


class TimeHorizon(str, Enum):
    now = "now"
    today = "today"
    this_week = "this_week"
    later = "later"


class MoodCheckIn(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    mood: Mood
    social_battery: SocialBattery
    intent: ConnectionIntent
    time_horizon: TimeHorizon
    is_open: bool = True
    online_only: bool = False
    neighborhood_radius_km: Optional[float] = 2.0
    interests: list[str] = Field(default_factory=list)
    conversation_style: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)


class CandidateMatch(BaseModel):
    user_id: str
    display_name: str
    compatibility_score: int
    serendipity_bonus: int
    total_score: int
    mood: Mood
    intent: ConnectionIntent
    time_horizon: TimeHorizon
    social_battery: SocialBattery
    shared_interests: list[str] = Field(default_factory=list)


class MediatorSessionCreate(BaseModel):
    requester_user_id: str
    candidate_user_id: str


class MediatorTurnCreate(BaseModel):
    speaker_user_id: str
    message: str


class MediatorMessage(BaseModel):
    id: int
    speaker_role: str
    speaker_user_id: Optional[str] = None
    speaker_display_name: str
    message: str
    created_at: datetime


class MediatorSessionSummary(BaseModel):
    session_id: int
    stage: int
    status: str
    direct_ready: bool
    participant_one: str
    participant_two: str
    participant_one_name: str
    participant_two_name: str
    shared_interests: list[str] = Field(default_factory=list)
    updated_at: datetime
    last_message_preview: Optional[str] = None


class MediatorSessionDetail(MediatorSessionSummary):
    handoff_message: Optional[str] = None
    messages: list[MediatorMessage] = Field(default_factory=list)


def normalize_interests(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = value.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)

    return cleaned[:8]


def shared_interests_for(user1: MoodCheckIn, user2: MoodCheckIn) -> list[str]:
    user2_interests = set(normalize_interests(user2.interests))
    shared = [item for item in normalize_interests(user1.interests) if item in user2_interests]
    return shared[:4]
