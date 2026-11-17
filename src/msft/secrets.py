"""Encrypted values for `.env`: plain variable names, `enc:v1:` values.

The key below is obfuscation, not secrecy: anyone with this repo can decrypt the file.
Its job is keeping a token out of git, logs, screenshots and shell history while still
letting a copied `.env` work on another machine. Set ``MSFT_SECRET_KEY`` to a private
Fernet key when that is not enough.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from msft.config import MsftError

ENV_PREFIX = "enc:v1:"
KEY_ENV = "MSFT_SECRET_KEY"
FILE_ENV = "MSFT_ENV_FILE"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Obfuscation key, not a secret: anyone with the repo can decrypt .env.
DEFAULT_KEY = "VtHvk0dodI4UdTWnkMrBVHOxEqLvBqvt8UZQXLj6C58="


def _fernet():
    """Build the Fernet cipher from ``MSFT_SECRET_KEY`` or the checked-in default."""
    from cryptography.fernet import Fernet

    key = os.environ.get(KEY_ENV) or DEFAULT_KEY
    try:
        return Fernet(key.encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise MsftError(f"{KEY_ENV} is not a valid Fernet key") from exc


def encrypt(value: str) -> str:
    """Wrap one plaintext value in the ``enc:v1:`` envelope."""
    if not isinstance(value, str):
        raise MsftError("only string values can be encrypted")
    try:
        token = _fernet().encrypt(value.encode("utf-8")).decode("ascii")
    except MsftError:
        raise
    except Exception as exc:  # noqa: BLE001 - cryptography raises several types
        raise MsftError("could not encrypt a .env value") from exc
    return f"{ENV_PREFIX}{token}"


def decrypt(value: str, name: str | None = None) -> str:
    """Decrypt an ``enc:v1:`` value; plaintext passes through unchanged.

    The error names the variable but never echoes the value or its length.
    """
    if not value.startswith(ENV_PREFIX):
        if value.startswith("enc:"):
            label = name or "a value"
            raise MsftError(f"{label} in .env uses an unsupported encryption envelope")
        return value
    token = value[len(ENV_PREFIX) :]
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - InvalidToken, bad padding, bad utf-8
        label = name or "value"
        raise MsftError(
            f"cannot decrypt {label} in .env; check {KEY_ENV} or re-run `msft secrets encrypt`"
        ) from exc


def resolve_env_path(path: str | Path | None = None) -> Path:
    """Resolve the `.env` path: explicit, then ``MSFT_ENV_FILE``, then the repo root."""
    if path is not None:
        return Path(path).expanduser()
    explicit = os.environ.get(FILE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    return REPO_ROOT / ".env"


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def read_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Parse a `.env` file into raw values, without decrypting or mutating the environment."""
    file = resolve_env_path(path)
    values: dict[str, str] = {}
    if not file.is_file():
        return values
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        name, sep, raw = stripped.partition("=")
        if not sep:
            continue
        name = name.strip()
        if name:
            values[name] = _strip_quotes(raw.strip())
    return values


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Load `.env` into ``os.environ``; a real exported variable always wins."""
    file = resolve_env_path(path)
    resolved: dict[str, str] = {}
    for name, raw in read_env_file(file).items():
        value = decrypt(raw, name)
        resolved[name] = value
        os.environ.setdefault(name, value)
    return resolved


def write_env_values(path: str | Path, updates: dict[str, str]) -> dict[str, str]:
    """Merge ``updates`` into a `.env`, preserving comments and unrelated lines.

    Returns ``{name: "added"|"updated"}``. The file is written atomically with mode 0600.
    """
    file = Path(path).expanduser()
    file.parent.mkdir(parents=True, exist_ok=True)
    lines = file.read_text(encoding="utf-8").splitlines() if file.is_file() else []
    pending = dict(updates)
    actions: dict[str, str] = {}
    written: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            written.append(line)
            continue
        body = stripped[len("export ") :].lstrip() if stripped.startswith("export ") else stripped
        name = body.partition("=")[0].strip()
        if name in pending:
            written.append(f"{name}={pending.pop(name)}")
            actions.setdefault(name, "updated")
        else:
            written.append(line)
    for name, value in pending.items():
        written.append(f"{name}={value}")
        actions[name] = "added"
    tmp = file.with_name(f".{file.name}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(written) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, file)
    finally:
        tmp.unlink(missing_ok=True)
    return actions


def assert_safe_output(path: str | Path) -> None:
    """Refuse to write a secret file that git tracks or would not ignore."""
    if shutil.which("git") is None:
        return
    file = Path(path).expanduser().resolve()
    parent = file.parent
    probe = subprocess.run(
        ["git", "-C", str(parent), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return
    tracked = subprocess.run(
        ["git", "-C", str(parent), "ls-files", "--error-unmatch", "--", str(file)],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0:
        raise MsftError(f"refusing to write {file}: git tracks it")
    ignored = subprocess.run(
        ["git", "-C", str(parent), "check-ignore", "-q", "--", str(file)],
        check=False,
    )
    if ignored.returncode != 0:
        raise MsftError(f"refusing to write {file}: it is not git-ignored")
