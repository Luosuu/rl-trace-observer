import re
from pathlib import Path

import pytest
from special_sanity.check_pr_description import is_unedited
from special_sanity.check_pr_title import ALLOWED_MODULES, InvalidTitleError, check_title

TEMPLATE = Path(__file__).resolve().parent.parent / ".github" / "PULL_REQUEST_TEMPLATE.md"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("[ci] chore: add pull request template", (["ci"], "chore", False)),
        ("[rl-insight, verl] feat: capture worker spans", (["rl-insight", "verl"], "feat", False)),
        ("[BREAKING][merger, manifest] feat: read the manifest", (["merger", "manifest"], "feat", True)),
        ("[1/N][merger] feat: add JSONL reader", (["merger"], "feat", False)),
        ("[2/3] [BREAKING] [Ray] Fix: runtime env", (["ray"], "fix", True)),
    ],
)
def test_valid_titles(title, expected):
    assert check_title(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Capture worker RL-Insight spans",
        "[verl] capture worker spans",
        "[verl] feature: capture worker spans",
        "[fsdp] feat: unknown module",
        "[verl] feat:",
        "[] feat: no module",
    ],
)
def test_invalid_titles(title):
    with pytest.raises(InvalidTitleError):
        check_title(title)


def test_template_lists_exactly_the_allowed_modules():
    template = TEMPLATE.read_text(encoding="utf-8")
    modules_line = next(line for line in template.splitlines() if "`{modules}` include" in line)
    assert tuple(re.findall(r"`([a-z0-9_-]+)`", modules_line)) == ALLOWED_MODULES


def test_unedited_description_is_detected():
    template = TEMPLATE.read_text(encoding="utf-8")

    assert is_unedited(template, template)
    assert not is_unedited("### What does this PR do?\n\nAdds a reader.\n", template)
