import json
import shutil
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wouldtheyhavemet.doc_governance import sync_repo, trigger_repo


class DocGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1] / "tests" / "_doc_governance_case"
        shutil.rmtree(self.repo_root, ignore_errors=True)
        (self.repo_root / "docs").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "research").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "docs" / "workflow.md").write_text("# Workflow\n", encoding="utf-8")
        (self.repo_root / "docs" / "roadmap.md").write_text("# Roadmap\n", encoding="utf-8")
        (self.repo_root / "docs" / "vision.md").write_text("# Vision\n", encoding="utf-8")

        config = {
            "version": 1,
            "triggers": {
                "planning_cycle": {
                    "label": "Planning cycle",
                    "description": "Activate during planning.",
                }
            },
            "tracked_docs": [
                {
                    "key": "workflow",
                    "title": "Workflow",
                    "path": "docs/workflow.md",
                    "update_when": "each planning cycle",
                    "triggers": ["planning_cycle"],
                },
                {
                    "key": "roadmap",
                    "title": "Roadmap",
                    "path": "docs/roadmap.md",
                    "update_when": "each planning cycle",
                    "triggers": ["planning_cycle"],
                },
                {
                    "key": "vision",
                    "title": "Vision",
                    "path": "docs/vision.md",
                    "update_when": "mission changes",
                    "triggers": [],
                },
            ],
        }
        (self.repo_root / "docs" / "doc-governance.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root, ignore_errors=True)

    def test_sync_creates_state_and_audit_files(self) -> None:
        result = sync_repo(self.repo_root, now=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc))

        self.assertTrue((self.repo_root / "docs" / "audit-log.md").exists())
        self.assertTrue((self.repo_root / "docs" / "review-queue.md").exists())
        self.assertTrue((self.repo_root / "docs" / ".doc-governance-state.json").exists())
        self.assertEqual(len(result["changes"]), 3)

        state = json.loads((self.repo_root / "docs" / ".doc-governance-state.json").read_text(encoding="utf-8"))
        self.assertIn("docs/workflow.md", state["tracked_docs"])

    def test_trigger_keeps_docs_pending_until_content_changes(self) -> None:
        baseline_time = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)
        trigger_time = baseline_time + timedelta(minutes=5)
        updated_time = trigger_time + timedelta(minutes=5)

        sync_repo(self.repo_root, now=baseline_time)
        trigger_repo(self.repo_root, "planning_cycle", note="weekly planning", now=trigger_time)

        queue = (self.repo_root / "docs" / "review-queue.md").read_text(encoding="utf-8")
        self.assertIn("docs/workflow.md", queue)
        self.assertIn("docs/roadmap.md", queue)

        (self.repo_root / "docs" / "workflow.md").write_text("# Workflow\n\nUpdated\n", encoding="utf-8")
        sync_repo(self.repo_root, now=updated_time)

        queue = (self.repo_root / "docs" / "review-queue.md").read_text(encoding="utf-8")
        self.assertNotIn("docs/workflow.md", queue)
        self.assertIn("docs/roadmap.md", queue)


if __name__ == "__main__":
    unittest.main()
