from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.database import (
    add_session_message,
    get_checkin,
    get_or_create_session,
    get_session,
    list_session_messages,
    list_sessions_for_user,
    update_session_state,
)
from api.models import (
    MediatorMessage,
    MediatorSessionCreate,
    MediatorSessionDetail,
    MediatorSessionSummary,
    MediatorTurnCreate,
    MoodCheckIn,
    shared_interests_for,
)

router = APIRouter()


@router.post("/mediator/sessions", response_model=MediatorSessionDetail)
def create_mediator_session(payload: MediatorSessionCreate) -> MediatorSessionDetail:
    requester = _load_checkin(payload.requester_user_id)
    candidate = _load_checkin(payload.candidate_user_id)

    if requester.user_id == candidate.user_id:
        raise HTTPException(status_code=400, detail="Mediator session requires two different users")

    session, created = get_or_create_session(requester.user_id, candidate.user_id, requester.user_id)
    if created:
        add_session_message(
            session["id"],
            speaker_role="mediator",
            message=_build_intro_message(requester, candidate),
        )

    return _build_session_detail(session["id"])


@router.get("/mediator/sessions/user/{user_id}", response_model=list[MediatorSessionSummary])
def list_user_sessions(user_id: str) -> list[MediatorSessionSummary]:
    _load_checkin(user_id)
    sessions = [_build_session_summary(row) for row in list_sessions_for_user(user_id)]
    return sessions


@router.get("/mediator/sessions/{session_id}", response_model=MediatorSessionDetail)
def get_mediator_session(session_id: int) -> MediatorSessionDetail:
    return _build_session_detail(session_id)


@router.post("/mediator/sessions/{session_id}/turn", response_model=MediatorSessionDetail)
def add_mediator_turn(session_id: int, payload: MediatorTurnCreate) -> MediatorSessionDetail:
    session = _load_session(session_id)
    speaker = _load_checkin(payload.speaker_user_id)
    participants = {session["participant_one"], session["participant_two"]}

    if speaker.user_id not in participants:
        raise HTTPException(status_code=403, detail="Only session participants can post messages")

    message_text = payload.message.strip()
    if not message_text:
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    add_session_message(
        session_id,
        speaker_role="participant",
        speaker_user_id=speaker.user_id,
        message=message_text,
    )

    stage, status, direct_ready = _derive_session_state(session_id)
    update_session_state(session_id, stage=stage, status=status, direct_ready=direct_ready)

    requester = _load_checkin(session["participant_one"])
    candidate = _load_checkin(session["participant_two"])
    add_session_message(
        session_id,
        speaker_role="mediator",
        message=_build_follow_up_message(session_id, requester, candidate, direct_ready),
    )

    return _build_session_detail(session_id)


def _load_checkin(user_id: str) -> MoodCheckIn:
    payload = get_checkin(user_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No check-in found")
    return MoodCheckIn(**payload)


def _load_session(session_id: int) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Mediator session not found")
    return session


def _derive_session_state(session_id: int) -> tuple[int, str, bool]:
    messages = list_session_messages(session_id)
    participant_messages = [message for message in messages if message["speaker_role"] == "participant"]
    speakers = {message["speaker_user_id"] for message in participant_messages if message["speaker_user_id"]}
    direct_ready = len(participant_messages) >= 4 and len(speakers) == 2

    if direct_ready:
        return 3, "handoff_ready", True
    if len(speakers) == 2:
        return 2, "bridging", False
    return 1, "active", False


def _build_intro_message(user1: MoodCheckIn, user2: MoodCheckIn) -> str:
    shared_interests = shared_interests_for(user1, user2)
    shared_interest_text = ", ".join(shared_interests) if shared_interests else "a calmer kind of connection"

    return (
        f"I'll open this gently. {user1.display_name or user1.user_id} and "
        f"{user2.display_name or user2.user_id} are both showing up for {user1.intent.value} "
        f"with a {user1.time_horizon.value.replace('_', ' ')} horizon. "
        f"Your clearest shared ground right now is {shared_interest_text}. "
        "Each of you can answer one easy prompt: what kind of low-pressure conversation or plan "
        "would feel good this week?"
    )


def _build_follow_up_message(
    session_id: int,
    user1: MoodCheckIn,
    user2: MoodCheckIn,
    direct_ready: bool,
) -> str:
    messages = list_session_messages(session_id)
    participant_messages = [message for message in messages if message["speaker_role"] == "participant"]
    participant_count = len({message["speaker_user_id"] for message in participant_messages if message["speaker_user_id"]})
    shared_interests = shared_interests_for(user1, user2)
    shared_interest_text = ", ".join(shared_interests) if shared_interests else "your shared preference for something low-pressure"

    if direct_ready:
        return (
            "You have enough context for a softer handoff now. I can step back after one direct note. "
            "Suggested opener: 'I liked how grounded this felt. Want to continue directly and compare what "
            "kind of plan would feel easiest first?'"
        )

    if participant_count == 1:
        return (
            "Thanks. I have one side of the intro. Once the other person answers, I'll bridge the overlap "
            "and suggest the smallest next step."
        )

    last_two = participant_messages[-2:]
    themes = _extract_short_theme(last_two)

    return (
        f"I'm hearing common ground around {themes} and {shared_interest_text}. "
        "Next prompt for both of you: share one boundary or pacing preference so the next step stays calm and safe."
    )


def _extract_short_theme(messages: list[dict]) -> str:
    recent_text = " ".join(message["message"].lower() for message in messages)
    theme_keywords = {
        "coffee": "meeting over coffee",
        "walk": "a simple walk",
        "book": "books",
        "music": "music",
        "movie": "movies",
        "quiet": "quiet energy",
        "slow": "slower pacing",
        "online": "starting online",
    }

    matches = [label for keyword, label in theme_keywords.items() if keyword in recent_text]
    if matches:
        return matches[0]
    return "taking things thoughtfully"


def _build_session_summary(session: dict) -> MediatorSessionSummary:
    session_id = session["id"]
    user1 = _load_checkin(session["participant_one"])
    user2 = _load_checkin(session["participant_two"])
    messages = list_session_messages(session_id)
    last_message_preview = messages[-1]["message"][:120] if messages else None

    return MediatorSessionSummary(
        session_id=session_id,
        stage=session["stage"],
        status=session["status"],
        direct_ready=bool(session["direct_ready"]),
        participant_one=user1.user_id,
        participant_two=user2.user_id,
        participant_one_name=user1.display_name or user1.user_id,
        participant_two_name=user2.display_name or user2.user_id,
        shared_interests=shared_interests_for(user1, user2),
        updated_at=session["updated_at"],
        last_message_preview=last_message_preview,
    )


def _build_session_detail(session_id: int) -> MediatorSessionDetail:
    session = _load_session(session_id)
    summary = _build_session_summary(session)
    user_lookup = {
        summary.participant_one: summary.participant_one_name,
        summary.participant_two: summary.participant_two_name,
    }

    messages = [
        MediatorMessage(
            id=row["id"],
            speaker_role=row["speaker_role"],
            speaker_user_id=row["speaker_user_id"],
            speaker_display_name=user_lookup.get(row["speaker_user_id"], "Mediator"),
            message=row["message"],
            created_at=row["created_at"],
        )
        for row in list_session_messages(session_id)
    ]

    handoff_message = None
    if summary.direct_ready:
        handoff_message = (
            "You're ready for a direct intro. Keep it light, confirm pace, and avoid pushing for specific location details too early."
        )

    return MediatorSessionDetail(
        **summary.model_dump(),
        handoff_message=handoff_message,
        messages=messages,
    )
