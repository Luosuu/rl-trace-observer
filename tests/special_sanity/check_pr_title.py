"""Check that a PR title follows `[BREAKING][{modules}] {type}: {description}`.

Run in CI with the title in ``PR_TITLE``. The allowed modules must match the
list in ``.github/PULL_REQUEST_TEMPLATE.md``.
"""

import os
import re
import sys

ALLOWED_MODULES = (
    "rl-insight",
    "verl",
    "tokenspeed",
    "viztracer",
    "torch",
    "ray",
    "merger",
    "manifest",
    "clock",
    "deps",
    "ci",
    "doc",
    "misc",
)
ALLOWED_TYPES = ("feat", "fix", "refactor", "chore", "test")

_PROGRESS = re.compile(r"^\[\d+/(\d+|N)\]\s*", re.IGNORECASE)
_BREAKING = re.compile(r"^\[BREAKING\]\s*", re.IGNORECASE)
_TITLE = re.compile(r"^\[(?P<modules>[a-z0-9_,\s-]+)\]\s+(?P<type>[a-z]+):\s+\S.*$", re.IGNORECASE)


class InvalidTitleError(ValueError):
    pass


def check_title(title: str) -> tuple[list[str], str, bool]:
    """Return ``(modules, type, is_breaking)`` or raise ``InvalidTitleError``."""
    core = _PROGRESS.sub("", title.strip(), count=1)
    is_breaking = bool(_BREAKING.match(core))
    core = _BREAKING.sub("", core, count=1)

    match = _TITLE.match(core)
    if match is None:
        raise InvalidTitleError(
            f"Invalid PR title: {title!r}. Expected format: [BREAKING][module, ...] type: description"
        )

    modules = [module.strip().lower() for module in match["modules"].split(",")]
    invalid_modules = [module for module in modules if module not in ALLOWED_MODULES]
    if invalid_modules:
        raise InvalidTitleError(
            f"Invalid modules: {', '.join(invalid_modules)}. Allowed modules: {', '.join(ALLOWED_MODULES)}"
        )

    change_type = match["type"].lower()
    if change_type not in ALLOWED_TYPES:
        raise InvalidTitleError(f"Invalid type: {change_type}. Allowed types: {', '.join(ALLOWED_TYPES)}")
    return modules, change_type, is_breaking


def main() -> int:
    title = os.environ.get("PR_TITLE", "")
    try:
        modules, change_type, is_breaking = check_title(title)
    except InvalidTitleError as error:
        print(f"❌ {error}")
        return 1
    breaking = " (BREAKING CHANGE)" if is_breaking else ""
    print(f"✅ PR title is valid: {title}, modules: {modules}, type: {change_type}{breaking}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
