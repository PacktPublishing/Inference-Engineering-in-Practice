import re
from pathlib import Path
from urllib.parse import unquote, urlparse

REPOSITORY_ROOT = Path(__file__).parents[3]
LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


def test_relative_markdown_links_resolve() -> None:
    failures = []
    markdown_paths = []
    for directory, subdirectories, filenames in REPOSITORY_ROOT.walk():
        subdirectories[:] = [
            name for name in subdirectories
            if not name.startswith(".") and name not in {"__pycache__", "node_modules"}
        ]
        markdown_paths.extend(directory / name for name in filenames if name.endswith(".md"))
    for markdown_path in markdown_paths:
        markdown = markdown_path.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(markdown):
            target = raw_target.strip().split(maxsplit=1)[0]
            parsed = urlparse(target)
            if parsed.scheme or target.startswith(("#", "mailto:")):
                continue

            relative = unquote(target.split("#", maxsplit=1)[0])
            resolved = (markdown_path.parent / relative).resolve()
            if not resolved.exists():
                failures.append(f"{markdown_path.relative_to(REPOSITORY_ROOT)} -> {target}")

    assert not failures, "\n".join(failures)
