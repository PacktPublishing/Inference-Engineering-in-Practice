import subprocess
from pathlib import Path

from scripts import check_python


def _environment(tmp_path: Path) -> Path:
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    return environment


def test_offloaded_environment_fails_before_starting_python(tmp_path, monkeypatch, capsys):
    environment = _environment(tmp_path)
    monkeypatch.setattr("sys.argv", ["check_python", "--environment", str(environment)])
    monkeypatch.setattr(check_python, "offloaded_files", lambda _: [environment / "lib.py"])

    def unexpected_import(*args, **kwargs):
        raise AssertionError("must not launch Python when dependencies are offloaded")

    monkeypatch.setattr(check_python.subprocess, "run", unexpected_import)
    assert check_python.main() == 1
    assert "offloaded files" in capsys.readouterr().out


def test_slow_import_is_bounded_and_reported(tmp_path, monkeypatch, capsys):
    environment = _environment(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["check_python", "--environment", str(environment), "--timeout", "0.1"]
    )
    monkeypatch.setattr(check_python, "offloaded_files", lambda _: [])

    def timeout(command, *, timeout):
        assert timeout == 0.1
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(check_python.subprocess, "run", timeout)
    assert check_python.main() == 1
    assert "Imports exceeded 0.1s" in capsys.readouterr().out


def test_symlinked_environment_scan_does_not_read_file_contents(tmp_path):
    environment = _environment(tmp_path)
    link = tmp_path / ".venv"
    link.symlink_to(environment, target_is_directory=True)
    assert check_python.offloaded_files(link.resolve()) == []
