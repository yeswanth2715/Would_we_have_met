from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

STATE_VERSION = 1
DEFAULT_CONFIG_PATH = Path("docs/doc-governance.yaml")
DEFAULT_STATE_PATH = Path("docs/.doc-governance-state.json")
DEFAULT_AUDIT_PATH = Path("docs/audit-log.md")
DEFAULT_QUEUE_PATH = Path("docs/review-queue.md")
HISTORY_LIMIT = 250


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_timestamp(value: str | None) -> str:
    parsed = parse_iso(value)
    if parsed is None:
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_sha(value: str | None) -> str:
    if not value:
        return "-"
    return value[:10]


def load_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    triggers = raw.get("triggers", {})
    tracked_docs = raw.get("tracked_docs", [])

    normalized_docs: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in tracked_docs:
        path = Path(item["path"]).as_posix()
        if path in seen_paths:
            raise ValueError(f"Duplicate tracked doc path in governance config: {path}")
        seen_paths.add(path)
        normalized = {
            "key": item["key"],
            "title": item["title"],
            "path": path,
            "update_when": item["update_when"],
            "triggers": [trigger for trigger in item.get("triggers", [])],
        }
        unknown_triggers = [trigger for trigger in normalized["triggers"] if trigger not in triggers]
        if unknown_triggers:
            raise ValueError(f"Unknown trigger(s) for {path}: {', '.join(unknown_triggers)}")
        normalized_docs.append(normalized)

    return {
        "version": raw.get("version", 1),
        "triggers": triggers,
        "tracked_docs": normalized_docs,
    }


def default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "last_synced_at": None,
        "tracked_docs": {},
        "history": [],
        "active_triggers": {},
        "trigger_events": [],
    }


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return default_state()

    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    state = default_state()
    state.update(loaded)
    state["tracked_docs"] = dict(state.get("tracked_docs", {}))
    state["history"] = list(state.get("history", []))
    state["active_triggers"] = dict(state.get("active_triggers", {}))
    state["trigger_events"] = list(state.get("trigger_events", []))
    return state


def write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def write_text_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def collect_snapshot(repo_root: Path, tracked_docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for doc in tracked_docs:
        rel_path = doc["path"]
        abs_path = repo_root / rel_path
        exists = abs_path.exists()
        modified_at = None
        sha256 = None
        size_bytes = None
        if exists:
            stat = abs_path.stat()
            modified_at = to_iso(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
            sha256 = sha256_file(abs_path)
            size_bytes = stat.st_size
        snapshot[rel_path] = {
            "title": doc["title"],
            "path": rel_path,
            "exists": exists,
            "modified_at": modified_at,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return snapshot


def detect_change(previous: dict[str, Any] | None, current: dict[str, Any]) -> str | None:
    if previous is None:
        return "baseline" if current["exists"] else "baseline_missing"
    if not previous.get("exists") and current["exists"]:
        return "created"
    if previous.get("exists") and not current["exists"]:
        return "missing"
    if previous.get("sha256") != current.get("sha256"):
        return "modified"
    return None


def append_change_history(
    state: dict[str, Any],
    doc: dict[str, Any],
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    event: str,
    now: datetime,
) -> None:
    state["history"].append(
        {
            "detected_at": to_iso(now),
            "path": doc["path"],
            "title": doc["title"],
            "event": event,
            "modified_at": current.get("modified_at"),
            "old_sha256": previous.get("sha256") if previous else None,
            "new_sha256": current.get("sha256"),
        }
    )
    state["history"] = state["history"][-HISTORY_LIMIT:]


def sync_state(config: dict[str, Any], state: dict[str, Any], repo_root: Path, now: datetime) -> list[dict[str, Any]]:
    snapshot = collect_snapshot(repo_root, config["tracked_docs"])
    changes: list[dict[str, Any]] = []

    for doc in config["tracked_docs"]:
        rel_path = doc["path"]
        current = snapshot[rel_path]
        previous = state["tracked_docs"].get(rel_path)
        event = detect_change(previous, current)

        if event is not None:
            append_change_history(state, doc, previous, current, event, now)
            changes.append({"path": rel_path, "event": event})

        change_count = 0 if previous is None else int(previous.get("change_count", 0))
        if event in {"created", "modified", "missing"}:
            change_count += 1

        last_detected_change_at = previous.get("last_detected_change_at") if previous else None
        last_event = previous.get("last_event") if previous else None
        if event is not None:
            last_detected_change_at = to_iso(now)
            last_event = event

        state["tracked_docs"][rel_path] = {
            "title": doc["title"],
            "path": rel_path,
            "exists": current["exists"],
            "modified_at": current["modified_at"],
            "sha256": current["sha256"],
            "size_bytes": current["size_bytes"],
            "last_detected_change_at": last_detected_change_at,
            "last_event": last_event,
            "change_count": change_count,
            "update_when": doc["update_when"],
            "triggers": doc["triggers"],
        }

    state["last_synced_at"] = to_iso(now)
    return changes


def add_trigger_event(
    state: dict[str, Any],
    trigger_key: str,
    action: str,
    now: datetime,
    note: str = "",
    activated_at: str | None = None,
) -> None:
    state["trigger_events"].append(
        {
            "trigger": trigger_key,
            "action": action,
            "at": to_iso(now),
            "activated_at": activated_at,
            "note": note,
        }
    )
    state["trigger_events"] = state["trigger_events"][-HISTORY_LIMIT:]


def activate_trigger(
    config: dict[str, Any],
    state: dict[str, Any],
    trigger_key: str,
    now: datetime,
    note: str = "",
) -> None:
    if trigger_key not in config["triggers"]:
        raise ValueError(f"Unknown trigger: {trigger_key}")

    state["active_triggers"][trigger_key] = {
        "activated_at": to_iso(now),
        "note": note,
    }
    add_trigger_event(state, trigger_key, "activated", now, note=note, activated_at=to_iso(now))


def clear_trigger(config: dict[str, Any], state: dict[str, Any], trigger_key: str, now: datetime) -> bool:
    if trigger_key not in config["triggers"]:
        raise ValueError(f"Unknown trigger: {trigger_key}")
    cleared = state["active_triggers"].pop(trigger_key, None)
    if cleared is None:
        return False
    add_trigger_event(state, trigger_key, "cleared", now, note=cleared.get("note", ""), activated_at=cleared.get("activated_at"))
    return True


def build_due_map(config: dict[str, Any], state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    due_map: dict[str, dict[str, Any]] = {}
    for trigger_key, details in state["active_triggers"].items():
        activated_at = parse_iso(details["activated_at"])
        due_docs: list[dict[str, Any]] = []
        for doc in config["tracked_docs"]:
            if trigger_key not in doc["triggers"]:
                continue
            tracked = state["tracked_docs"].get(doc["path"], {})
            last_detected = parse_iso(tracked.get("last_detected_change_at"))
            if last_detected is None or activated_at is None or last_detected <= activated_at:
                due_docs.append(doc)
        due_map[trigger_key] = {
            "details": details,
            "due_docs": due_docs,
        }
    return due_map


def auto_resolve_triggers(config: dict[str, Any], state: dict[str, Any], now: datetime) -> None:
    due_map = build_due_map(config, state)
    resolved = [trigger_key for trigger_key, entry in due_map.items() if not entry["due_docs"]]
    for trigger_key in resolved:
        details = state["active_triggers"].pop(trigger_key)
        add_trigger_event(
            state,
            trigger_key,
            "resolved",
            now,
            note=details.get("note", ""),
            activated_at=details.get("activated_at"),
        )


def build_review_queue(config: dict[str, Any], state: dict[str, Any], now: datetime) -> str:
    due_map = build_due_map(config, state)
    lines = [
        "# Review Queue",
        "",
        "This file is auto-generated by `python scripts/docs_governance.py sync`.",
        "",
        f"Generated at: `{to_iso(now)}`",
        "",
    ]

    if not due_map:
        lines.extend(
            [
                "## Active Triggers",
                "",
                "No active review triggers.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Active Triggers",
            "",
            "| Trigger | Activated At | Note | Pending Docs |",
            "| --- | --- | --- | --- |",
        ]
    )

    for trigger_key, entry in due_map.items():
        trigger_label = config["triggers"][trigger_key]["label"]
        details = entry["details"]
        due_titles = ", ".join(doc["title"] for doc in entry["due_docs"]) or "None"
        note = details.get("note") or "-"
        lines.append(
            f"| {trigger_label} | {format_timestamp(details['activated_at'])} | {note} | {due_titles} |"
        )

    lines.extend(["", "## Pending Actions", ""])
    pending_any = False
    for trigger_key, entry in due_map.items():
        if not entry["due_docs"]:
            continue
        pending_any = True
        trigger_label = config["triggers"][trigger_key]["label"]
        lines.append(f"### {trigger_label}")
        lines.append("")
        for doc in entry["due_docs"]:
            lines.append(f"- `{doc['path']}`: update when {doc['update_when']}")
        lines.append("")

    if not pending_any:
        lines.append("All active triggers have been satisfied.")
        lines.append("")

    return "\n".join(lines) + "\n"


def doc_status(state_entry: dict[str, Any], due_triggers: list[str]) -> str:
    if not state_entry.get("exists"):
        return "missing"
    if due_triggers:
        joined = ", ".join(due_triggers)
        return f"pending review ({joined})"
    return "tracked"


def build_audit_log(config: dict[str, Any], state: dict[str, Any], now: datetime) -> str:
    due_map = build_due_map(config, state)
    due_by_path: dict[str, list[str]] = {}
    for trigger_key, entry in due_map.items():
        label = config["triggers"][trigger_key]["label"]
        for doc in entry["due_docs"]:
            due_by_path.setdefault(doc["path"], []).append(label)

    lines = [
        "# Docs Audit Log",
        "",
        "This file is auto-generated by `python scripts/docs_governance.py sync`.",
        "",
        f"Generated at: `{to_iso(now)}`",
        f"Last sync: `{state.get('last_synced_at') or '-'}`",
        "",
        "## Tracked Documents",
        "",
        "| Document | Path | Last Modified | Last Detected Change | Hash | Status | Update Rule |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for doc in config["tracked_docs"]:
        entry = state["tracked_docs"].get(doc["path"], {})
        due_triggers = due_by_path.get(doc["path"], [])
        lines.append(
            "| {title} | `{path}` | {modified} | {changed} | `{sha}` | {status} | {rule} |".format(
                title=doc["title"],
                path=doc["path"],
                modified=format_timestamp(entry.get("modified_at")),
                changed=format_timestamp(entry.get("last_detected_change_at")),
                sha=short_sha(entry.get("sha256")),
                status=doc_status(entry, due_triggers),
                rule=doc["update_when"],
            )
        )

    lines.extend(["", "## Active Review Triggers", ""])
    if due_map:
        lines.extend(
            [
                "| Trigger | Activated At | Note | Pending Docs |",
                "| --- | --- | --- | --- |",
            ]
        )
        for trigger_key, entry in due_map.items():
            details = entry["details"]
            trigger_label = config["triggers"][trigger_key]["label"]
            pending = ", ".join(f"`{doc['path']}`" for doc in entry["due_docs"]) or "None"
            note = details.get("note") or "-"
            lines.append(
                f"| {trigger_label} | {format_timestamp(details['activated_at'])} | {note} | {pending} |"
            )
    else:
        lines.append("No active review triggers.")

    lines.extend(["", "## Recent Trigger Activity", ""])
    trigger_events = list(reversed(state.get("trigger_events", [])[-12:]))
    if trigger_events:
        for event in trigger_events:
            label = config["triggers"].get(event["trigger"], {}).get("label", event["trigger"])
            note = f" Note: {event['note']}" if event.get("note") else ""
            activated = (
                f" Trigger opened at {format_timestamp(event['activated_at'])}."
                if event.get("activated_at")
                else ""
            )
            lines.append(f"- {format_timestamp(event['at'])}: {label} was {event['action']}.{activated}{note}")
    else:
        lines.append("No trigger activity recorded yet.")

    lines.extend(["", "## Recent Document Changes", ""])
    history = list(reversed(state.get("history", [])[-20:]))
    if history:
        for item in history:
            sha = short_sha(item.get("new_sha256"))
            modified = format_timestamp(item.get("modified_at"))
            lines.append(
                f"- {format_timestamp(item['detected_at'])}: `{item['path']}` -> {item['event']} (modified {modified}, sha `{sha}`)"
            )
    else:
        lines.append("No document changes recorded yet.")

    return "\n".join(lines) + "\n"


def sync_repo(
    repo_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    config = load_config(repo_root / config_path)
    state = load_state(repo_root / state_path)
    changes = sync_state(config, state, repo_root, now)
    auto_resolve_triggers(config, state, now)

    audit_content = build_audit_log(config, state, now)
    queue_content = build_review_queue(config, state, now)

    files_written = {
        "state": write_json_if_changed(repo_root / state_path, state),
        "audit": write_text_if_changed(repo_root / audit_path, audit_content),
        "queue": write_text_if_changed(repo_root / queue_path, queue_content),
    }

    return {
        "changes": changes,
        "files_written": files_written,
        "active_triggers": state["active_triggers"],
    }


def trigger_repo(
    repo_root: Path,
    trigger_key: str,
    *,
    note: str = "",
    config_path: Path = DEFAULT_CONFIG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    config = load_config(repo_root / config_path)
    state = load_state(repo_root / state_path)
    sync_state(config, state, repo_root, now)
    activate_trigger(config, state, trigger_key, now, note=note)
    auto_resolve_triggers(config, state, now)

    audit_content = build_audit_log(config, state, now)
    queue_content = build_review_queue(config, state, now)

    files_written = {
        "state": write_json_if_changed(repo_root / state_path, state),
        "audit": write_text_if_changed(repo_root / audit_path, audit_content),
        "queue": write_text_if_changed(repo_root / queue_path, queue_content),
    }

    return {
        "files_written": files_written,
        "active_triggers": state["active_triggers"],
    }


def clear_repo_trigger(
    repo_root: Path,
    trigger_key: str,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    config = load_config(repo_root / config_path)
    state = load_state(repo_root / state_path)
    cleared = clear_trigger(config, state, trigger_key, now)

    audit_content = build_audit_log(config, state, now)
    queue_content = build_review_queue(config, state, now)

    files_written = {
        "state": write_json_if_changed(repo_root / state_path, state),
        "audit": write_text_if_changed(repo_root / audit_path, audit_content),
        "queue": write_text_if_changed(repo_root / queue_path, queue_content),
    }

    return {
        "cleared": cleared,
        "files_written": files_written,
        "active_triggers": state["active_triggers"],
    }


def print_sync_summary(result: dict[str, Any]) -> None:
    changes = result.get("changes", [])
    if not changes:
        print("Docs governance sync complete: no tracked document content changes detected.")
    else:
        print("Docs governance sync complete:")
        for change in changes:
            print(f"  - {change['path']}: {change['event']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track and audit the repo's product documents.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("sync", help="Refresh the docs audit log and review queue.")

    trigger_parser = subparsers.add_parser("trigger", help="Activate a docs review trigger.")
    trigger_parser.add_argument("trigger_key", help="Trigger key from docs/doc-governance.yaml.")
    trigger_parser.add_argument("--note", default="", help="Optional note for the trigger event.")

    clear_parser = subparsers.add_parser("clear-trigger", help="Clear an active docs review trigger.")
    clear_parser.add_argument("trigger_key", help="Trigger key from docs/doc-governance.yaml.")

    watch_parser = subparsers.add_parser("watch", help="Watch tracked docs and refresh the audit on change.")
    watch_parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds.")

    subparsers.add_parser("list-triggers", help="List available docs review triggers.")
    return parser


def list_triggers(repo_root: Path) -> int:
    config = load_config(repo_root / DEFAULT_CONFIG_PATH)
    print("Available docs review triggers:")
    for key, details in config["triggers"].items():
        print(f"  - {key}: {details['label']} ({details['description']})")
    return 0


def watch_repo(repo_root: Path, interval: float) -> int:
    print(f"Watching tracked docs every {interval:.1f}s. Press Ctrl+C to stop.")
    previous_signature: tuple[tuple[str, str | None], ...] | None = None
    try:
        while True:
            config = load_config(repo_root / DEFAULT_CONFIG_PATH)
            snapshot = collect_snapshot(repo_root, config["tracked_docs"])
            signature = tuple((path, details.get("sha256")) for path, details in sorted(snapshot.items()))
            if signature != previous_signature:
                result = sync_repo(repo_root)
                print_sync_summary(result)
                previous_signature = signature
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped docs governance watch mode.")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "sync"
    repo_root = Path(__file__).resolve().parents[2]

    if command == "sync":
        result = sync_repo(repo_root)
        print_sync_summary(result)
        return 0

    if command == "trigger":
        result = trigger_repo(repo_root, args.trigger_key, note=args.note)
        print(f"Activated trigger `{args.trigger_key}`. Active triggers: {', '.join(result['active_triggers']) or 'none'}")
        return 0

    if command == "clear-trigger":
        result = clear_repo_trigger(repo_root, args.trigger_key)
        if result["cleared"]:
            print(f"Cleared trigger `{args.trigger_key}`.")
        else:
            print(f"Trigger `{args.trigger_key}` was not active.")
        return 0

    if command == "watch":
        return watch_repo(repo_root, args.interval)

    if command == "list-triggers":
        return list_triggers(repo_root)

    parser.print_help()
    return 1
