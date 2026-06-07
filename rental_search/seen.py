import json
from pathlib import Path

SEEN_FILE = Path(__file__).parent / "seen.json"


def load_seen() -> set[str]:
    if not SEEN_FILE.exists():
        return set()
    return set(json.loads(SEEN_FILE.read_text()))


def save_seen(seen: set[str]) -> None:
    SEEN_FILE.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2))


def filter_new(listings: list[dict], seen: set[str]) -> tuple[list[dict], set[str]]:
    new = [l for l in listings if l["id"] not in seen]
    updated = seen | {l["id"] for l in listings}
    return new, updated
