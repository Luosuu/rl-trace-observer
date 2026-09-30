"""Check that the PR description replaced the template's opening placeholder.

Run in CI with ``GITHUB_EVENT_PATH`` pointing at the pull request event.
"""

import json
import os
import sys
from pathlib import Path

NUM_LINES = 5
TEMPLATE = Path(os.getenv("GITHUB_WORKSPACE", ".")) / ".github" / "PULL_REQUEST_TEMPLATE.md"


def first_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()[:NUM_LINES]]


def is_unedited(body: str, template: str) -> bool:
    return first_lines(body) == first_lines(template)


def main() -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    body = event.get("pull_request", {}).get("body") or ""
    if is_unedited(body, TEMPLATE.read_text(encoding="utf-8")):
        print(
            "❌ The '### What does this PR do?' section still has the template placeholder. "
            "Replace it with a concise description of this PR."
        )
        return 1
    print("✅ '### What does this PR do?' section has been filled out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
