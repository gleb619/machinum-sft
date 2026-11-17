"""Encrypted `.env` values: round-trip, loader precedence and the CLI's name-only output."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from msft import secrets
from msft.cli import EXIT_ERROR, EXIT_OK, main
from msft.config import MsftError


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read the developer's real `.env` or key while testing."""
    monkeypatch.delenv(secrets.KEY_ENV, raising=False)
    monkeypatch.setenv(secrets.FILE_ENV, str(tmp_path / "absent.env"))


def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text(".env\n", encoding="utf-8")
    return repo


def test_round_trip_and_plaintext_passthrough() -> None:
    envelope = secrets.encrypt("hf_secret_value")
    assert envelope.startswith(secrets.ENV_PREFIX)
    assert "hf_secret_value" not in envelope
    assert secrets.decrypt(envelope) == "hf_secret_value"
    assert secrets.decrypt("plain-value") == "plain-value"


def test_unknown_envelope_is_not_treated_as_plaintext() -> None:
    with pytest.raises(MsftError, match="unsupported encryption envelope"):
        secrets.decrypt("enc:v2:whatever", "HF_TOKEN")


def test_tampered_token_raises_without_leaking_the_value() -> None:
    envelope = secrets.encrypt("do-not-print-me")
    middle = len(envelope) // 2
    swap = "A" if envelope[middle] != "A" else "B"
    tampered = envelope[:middle] + swap + envelope[middle + 1 :]
    with pytest.raises(MsftError) as caught:
        secrets.decrypt(tampered, "HF_TOKEN")
    message = str(caught.value)
    assert "HF_TOKEN" in message
    assert "do-not-print-me" not in message


def test_loader_sets_variables_and_a_real_env_var_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nPLAIN='quoted value'\nSECRET=" + secrets.encrypt("from-file") + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SECRET", "from-shell")
    resolved = secrets.load_env_file(env_file)
    assert resolved["PLAIN"] == "quoted value"
    assert resolved["SECRET"] == "from-file"
    assert os.environ["PLAIN"] == "quoted value"
    assert os.environ["SECRET"] == "from-shell"


def test_write_env_values_merges_and_is_private(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("# keep\nOTHER=one\nHF_TOKEN=old\n", encoding="utf-8")
    actions = secrets.write_env_values(
        target, {"HF_TOKEN": secrets.encrypt("new"), "PLAIN_NAME": secrets.encrypt("v")}
    )
    text = target.read_text(encoding="utf-8")
    assert actions == {"HF_TOKEN": "updated", "PLAIN_NAME": "added"}
    assert "# keep" in text and "OTHER=one" in text
    assert "HF_TOKEN=enc:v1:" in text and "PLAIN_NAME=enc:v1:" in text
    assert "new" not in text.replace("enc:v1:", "")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_cli_encrypt_writes_names_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = git_repo(tmp_path)
    source = tmp_path / "secrets.json"
    source.write_text(json.dumps({"hf_token": "hf_test_123", "other": "second"}), encoding="utf-8")
    out = repo / ".env"
    out.write_text("# keep\nUNRELATED=one\nHF_TOKEN=old\n", encoding="utf-8")

    code = main(["secrets", "encrypt", "--from-json", str(source), "--out", str(out)])
    stdout = capsys.readouterr().out

    assert code == EXIT_OK
    text = out.read_text(encoding="utf-8")
    assert "HF_TOKEN=enc:v1:" in text and "OTHER=enc:v1:" in text
    assert "UNRELATED=one" in text and "# keep" in text
    assert "hf_test_123" not in text and "second" not in text
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert "HF_TOKEN" in stdout and "OTHER" in stdout
    assert "hf_test_123" not in stdout


def test_cli_encrypt_refuses_a_tracked_file(tmp_path: Path) -> None:
    repo = git_repo(tmp_path)
    source = tmp_path / "secrets.json"
    source.write_text(json.dumps({"hf_token": "hf_test_123"}), encoding="utf-8")
    tracked = repo / "tracked.env"
    tracked.write_text("HF_TOKEN=old\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.env"], cwd=repo, check=True)

    code = main(["secrets", "encrypt", "--from-json", str(source), "--out", str(tracked)])
    assert code == EXIT_ERROR
    assert tracked.read_text(encoding="utf-8") == "HF_TOKEN=old\n"


def test_cli_encrypt_refuses_an_unignored_file(tmp_path: Path) -> None:
    repo = git_repo(tmp_path)
    source = tmp_path / "secrets.json"
    source.write_text(json.dumps({"hf_token": "hf_test_123"}), encoding="utf-8")
    out = repo / "not-ignored.env"

    code = main(["secrets", "encrypt", "--from-json", str(source), "--out", str(out)])
    assert code == EXIT_ERROR
    assert not out.exists()


def test_cli_list_shows_state_and_never_a_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"PLAIN=hello\nSECRET={secrets.encrypt('s3cr3t')}\n", encoding="utf-8"
    )

    code = main(["secrets", "list", "--file", str(env_file)])
    stdout = capsys.readouterr().out

    assert code == EXIT_OK
    assert "PLAIN\tplain" in stdout
    assert "SECRET\tencrypted" in stdout
    assert "hello" not in stdout and "s3cr3t" not in stdout
