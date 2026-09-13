import re
import textwrap
from pathlib import Path

import pytest

CHAPTER_DIR = Path(__file__).parents[1]
CHAPTER_PATH = CHAPTER_DIR / "chapter.md"
SNIPPET_SOURCES = {
    "prompt-to-logits": "code/prompt_to_logits.py",
    "cached-loop": "code/generation_loop.py",
}


def _source_snippet(name: str, relative_path: str) -> str:
    source = (CHAPTER_DIR / relative_path).read_text(encoding="utf-8")
    match = re.search(
        rf"^[ \t]*# snippet:{re.escape(name)}\n(?P<body>.*?)^[ \t]*# end-snippet",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"source marker not found: {name}"
    return textwrap.dedent(match.group("body")).strip()


def _chapter_snippet(name: str) -> str:
    chapter = CHAPTER_PATH.read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- snippet:{re.escape(name)} -->\s*```python\n(?P<body>.*?)\n```",
        chapter,
        flags=re.DOTALL,
    )
    assert match is not None, f"chapter marker not found: {name}"
    return match.group("body").strip()


def test_chapter_snippets_match_executed_source() -> None:
    if not CHAPTER_PATH.exists():
        pytest.skip("chapter.md is not present")
    for name, relative_path in SNIPPET_SOURCES.items():
        assert _chapter_snippet(name) == _source_snippet(name, relative_path), name
