from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional

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
    mood: Mood
    social_battery: SocialBattery
    intent: ConnectionIntent
    time_horizon: TimeHorizon
    is_open: bool = True
    online_only: bool = False
    neighborhood_radius_km: Optional[float] = 2.0
    timestamp: datetime = datetime.utcnow()
