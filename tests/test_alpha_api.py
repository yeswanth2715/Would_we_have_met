import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.database import get_mood_history, get_mood_pattern, init_db
from api.main import app


class AlphaApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="wwhm_alpha_api_"))
        self.db_path = self.tmp_dir / "alpha.db"
        self.previous_db_path = os.environ.get("WWHM_DB_PATH")
        os.environ["WWHM_DB_PATH"] = str(self.db_path)
        init_db()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        if self.previous_db_path is None:
            os.environ.pop("WWHM_DB_PATH", None)
        else:
            os.environ["WWHM_DB_PATH"] = self.previous_db_path
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_root_serves_frontend(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Would We Have Met", response.text)
        self.assertIn("Mediated Intros", response.text)

    def test_checkins_persist_and_candidates_load(self) -> None:
        self._post_checkin(
            user_id="aanya",
            display_name="Aanya",
            mood="curious",
            social_battery="medium",
            intent="friendship",
            time_horizon="today",
            interests=["books", "coffee walks"],
        )
        self._post_checkin(
            user_id="mira",
            display_name="Mira",
            mood="open",
            social_battery="medium",
            intent="friendship",
            time_horizon="today",
            interests=["books", "film"],
        )

        stored = self.client.get("/checkin/mood/aanya")
        self.assertEqual(stored.status_code, 200)
        self.assertEqual(stored.json()["display_name"], "Aanya")

        candidates = self.client.get("/candidates/aanya")
        self.assertEqual(candidates.status_code, 200)
        body = candidates.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["user_id"], "mira")
        self.assertIn("books", body[0]["shared_interests"])

    def test_mediator_session_advances_to_handoff(self) -> None:
        self._post_checkin(
            user_id="noah",
            display_name="Noah",
            mood="reflective",
            social_battery="low",
            intent="conversation",
            time_horizon="this_week",
            interests=["music", "quiet walks"],
        )
        self._post_checkin(
            user_id="lena",
            display_name="Lena",
            mood="quiet",
            social_battery="low",
            intent="conversation",
            time_horizon="this_week",
            interests=["music", "coffee"],
        )

        created = self.client.post(
            "/mediator/sessions",
            json={"requester_user_id": "noah", "candidate_user_id": "lena"},
        )
        self.assertEqual(created.status_code, 200)
        session_id = created.json()["session_id"]

        for speaker, message in [
            ("noah", "A quiet coffee walk sounds good to me."),
            ("lena", "I like starting with a short online chat first."),
            ("noah", "Slow pacing works well for me too."),
            ("lena", "Same here. I prefer a low-pressure start."),
        ]:
            response = self.client.post(
                f"/mediator/sessions/{session_id}/turn",
                json={"speaker_user_id": speaker, "message": message},
            )
            self.assertEqual(response.status_code, 200)

        detail = self.client.get(f"/mediator/sessions/{session_id}")
        self.assertEqual(detail.status_code, 200)
        payload = detail.json()
        self.assertTrue(payload["direct_ready"])
        self.assertEqual(payload["status"], "handoff_ready")
        self.assertGreaterEqual(len(payload["messages"]), 5)
        self.assertIn("direct intro", payload["handoff_message"].lower())

    def test_checkin_updates_are_archived_in_mood_history(self) -> None:
        self._post_checkin(
            user_id="tara",
            display_name="Tara",
            mood="curious",
            social_battery="medium",
            intent="friendship",
            time_horizon="today",
            interests=["books"],
        )
        self._post_checkin(
            user_id="tara",
            display_name="Tara",
            mood="quiet",
            social_battery="low",
            intent="conversation",
            time_horizon="this_week",
            interests=["books", "music"],
        )
        self._post_checkin(
            user_id="tara",
            display_name="Tara",
            mood="energized",
            social_battery="high",
            intent="activity_partner",
            time_horizon="now",
            interests=["music"],
        )

        history = get_mood_history("tara")
        pattern = get_mood_pattern("tara")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["mood"], "quiet")
        self.assertEqual(history[1]["mood"], "curious")
        self.assertEqual(pattern, {"curious": 1, "quiet": 1})

    def _post_checkin(
        self,
        *,
        user_id: str,
        display_name: str,
        mood: str,
        social_battery: str,
        intent: str,
        time_horizon: str,
        interests: list[str],
    ) -> None:
        response = self.client.post(
            "/checkin/mood",
            json={
                "user_id": user_id,
                "display_name": display_name,
                "mood": mood,
                "social_battery": social_battery,
                "intent": intent,
                "time_horizon": time_horizon,
                "is_open": True,
                "online_only": False,
                "neighborhood_radius_km": 2.0,
                "interests": interests,
                "conversation_style": "low-pressure",
            },
        )

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
