import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

REPOSITORY_ROOT = Path(__file__).parents[1]
LINK_PATTERN = re.compile(r"!?\[[^\]]*]\((<[^>]+>|[^)]+)\)")


def test_relative_markdown_links_resolve() -> None:
    failures = []
    # Public documentation must work in a clean checkout. Ignored author drafts
    # and historical private archives are not CI inputs.
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=REPOSITORY_ROOT,
        )
        .decode()
        .split("\0")
    )
    markdown_paths = [
        REPOSITORY_ROOT / name
        for name in set(names)
        if name.endswith(".md") and (REPOSITORY_ROOT / name).is_file()
    ]
    for markdown_path in markdown_paths:
        markdown = markdown_path.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(markdown):
            target = (
                raw_target[1:-1]
                if raw_target.startswith("<")
                else raw_target.strip().split(maxsplit=1)[0]
            )
            parsed = urlparse(target)
            if parsed.scheme or target.startswith(("#", "mailto:")):
                continue

            relative = unquote(target.split("#", maxsplit=1)[0])
            resolved = (markdown_path.parent / relative).resolve()
            if not resolved.exists():
                failures.append(f"{markdown_path.relative_to(REPOSITORY_ROOT)} -> {target}")

    assert not failures, "\n".join(failures)
