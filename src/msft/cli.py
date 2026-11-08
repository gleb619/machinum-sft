"""Command-line parsing and dispatch for `msft`.

Only argparse is used here, and every implementation module is imported *inside* its handler,
so `python -m msft --help` stays usable on a machine without torch, unsloth or trl.

Exit codes: 0 success, 1 red eval gates, 2 user/config error (`error: ...` on stderr), 3 the
requested implementation module is not written yet. Every step that would train or download is a
printed plan unless `--execute` is passed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from msft import __version__
from msft.config import MsftError, RunConfig, dump_config, load_config, redact, validate_config

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NOT_IMPLEMENTED = 3

DEFAULT_CONFIG = Path("configs") / "qwen3.5-9b-qlora.yaml"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

Handler = Callable[[RunConfig | None, argparse.Namespace], int]


class _NotImplementedYet(Exception):
    """An implementation module (or one of its entry points) does not exist yet."""

    def __init__(self, module: str, detail: str) -> None:
        super().__init__(f"{module}: {detail}")
        self.module = module
        self.detail = detail

    @property
    def message(self) -> str:
        """Human-readable line naming the missing module."""
        return f"{self.module} is not implemented yet ({self.detail})"


# --- lazy imports ---------------------------------------------------------


def _impl(module: str) -> Any:
    """Import an implementation module lazily, raising `_NotImplementedYet` when absent."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        missing = getattr(exc, "name", None) == module
        detail = "module not found" if missing else f"import failed: {exc}"
        raise _NotImplementedYet(module, detail) from exc


def _func(module: Any, name: str) -> Callable[..., Any]:
    """Return a callable attribute of a lazily imported module, or fail as unimplemented."""
    attr = getattr(module, name, None)
    if not callable(attr):
        raise _NotImplementedYet(f"{module.__name__}.{name}", "function not found")
    return attr


# --- output helpers -------------------------------------------------------


def _log(message: str) -> None:
    """Write one redacted line to stderr; stdout is reserved for payloads."""
    print(redact(message), file=sys.stderr)


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, paths and mappings into JSON-serialisable builtins."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _emit(as_json: bool, payload: Any, text: str) -> None:
    """Print exactly one JSON object when `as_json` is set, otherwise the text report."""
    if as_json:
        print(json.dumps(_jsonable(payload), ensure_ascii=False, default=str))
    else:
        print(text)


def _scalar(value: Any) -> str:
    """Render a single value for the human-readable report."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "-"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_scalar(item) for item in value) or "-"
    if isinstance(value, dict):
        return json.dumps(_jsonable(value), ensure_ascii=False)
    return str(value)


def _render(title: str, data: dict[str, Any]) -> str:
    """Render a flat (optionally one level nested) mapping as an indented report."""
    lines = [f"{title}:"]
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            lines.extend(
                f"    {sub_key}: {_scalar(sub_value)}" for sub_key, sub_value in value.items()
            )
        else:
            lines.append(f"  {key}: {_scalar(value)}")
    return "\n".join(lines)


# --- configuration --------------------------------------------------------


def _default_config_path() -> Path | None:
    """Return the first existing default config, looked up from cwd and then the repo root."""
    for base in (Path.cwd(), REPO_ROOT):
        candidate = base / DEFAULT_CONFIG
        if candidate.is_file():
            return candidate
    return None


def _resolve_config(args: argparse.Namespace) -> RunConfig | None:
    """Load `--config`, else the default config, else return None."""
    explicit = getattr(args, "config", None)
    if explicit:
        return load_config(explicit)
    default = _default_config_path()
    if default is None:
        return None
    return load_config(default)


def _require_config(cfg: RunConfig | None) -> RunConfig:
    """Return the loaded config or explain how to supply one."""
    if cfg is None:
        raise MsftError(
            "no config given: pass --config PATH "
            f"(default {DEFAULT_CONFIG.as_posix()} was not found)"
        )
    return cfg


def _remote_host() -> str:
    """Read the ssh target from the environment; never hardcode a host."""
    host = os.environ.get("MSFT_REMOTE_HOST", "").strip()
    if not host:
        raise MsftError("MSFT_REMOTE_HOST is not set; export it or add it to your .env file")
    return host


# --- handlers -------------------------------------------------------------


def _cmd_doctor(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Probe the local environment (config optional) and print the report."""
    doctor = _impl("msft.doctor")
    checks = _func(doctor, "run_doctor")(cfg)
    if args.as_json:
        _emit(True, {"checks": checks}, "")
        return EXIT_OK
    print(_func(doctor, "format_report")(checks, False))
    return EXIT_OK


def _cmd_data_build(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Build, filter and split the dataset described by the config."""
    config = _require_config(cfg)
    data = _impl("msft.data")
    stats = _func(data, "build_dataset")(config, execute=bool(args.execute))
    payload = _jsonable(stats)
    _emit(args.as_json, payload, _render("dataset", payload))
    return EXIT_OK


def _model_stage(module: str, title: str, cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Shared body for train/merge/quantize: script, dry-run plan, or execution."""
    config = _require_config(cfg)
    stage = _impl(module)
    if getattr(args, "script", False):
        script = _func(stage, "build_script")(config)
        _emit(args.as_json, {"script": script}, str(script))
        return EXIT_OK
    if args.execute:
        result = _jsonable(_func(stage, "run")(config, execute=True))
        _emit(args.as_json, result, _render(title, result))
        return EXIT_OK
    # Dry run: a missing external tool (llama.cpp, a merged adapter) is a warning on the
    # plan, not an error — the whole point is to see what would run on this machine.
    try:
        plan = _jsonable(_func(stage, "plan")(config))
    except MsftError as exc:
        plan = _jsonable(_func(stage, "plan")(config, require_tools=False)) if _accepts_kwarg(
            stage, "plan", "require_tools"
        ) else {"error": redact(str(exc))}
    _emit(args.as_json, plan, _render(f"{title} plan", plan))
    return EXIT_OK


def _accepts_kwarg(module: Any, func_name: str, kwarg: str) -> bool:
    """True when the callable takes `kwarg`; used to degrade gracefully on dry runs."""
    import inspect

    try:
        signature = inspect.signature(_func(module, func_name))
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    return kwarg in signature.parameters


def _cmd_train(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Plan or execute QLoRA/SFT training."""
    return _model_stage("msft.train", "train", cfg, args)


def _cmd_merge(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Plan or execute adapter merging into 16-bit weights."""
    return _model_stage("msft.merge", "merge", cfg, args)


def _cmd_quantize(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Plan or execute GGUF conversion and quantisation."""
    return _model_stage("msft.quantize", "quantize", cfg, args)


def _cmd_eval(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Run the protocol gates against a predictions file."""
    evaluate = _impl("msft.evaluate")
    predictions = args.predictions
    if predictions is None:
        if cfg is None:
            raise MsftError("eval needs --predictions PATH (no config to derive one from)")
        predictions = str(cfg.output_path / "eval" / "predictions.jsonl")
    gates = _func(evaluate, "run_gates")(Path(predictions), as_json=False)
    rows = _jsonable(gates)
    passed = bool(rows) and all(bool(row.get("passed")) for row in rows)
    text = "\n".join(
        f"{'PASS' if row.get('passed') else 'FAIL'} {row.get('gate')}: {row.get('detail')}"
        for row in rows
    )
    text = f"{text}\ngates {'green' if passed else 'red'} ({len(rows)} checked)"
    _emit(args.as_json, {"gates": rows, "passed": passed}, text)
    return EXIT_OK if passed else 1


def _cmd_remote_preflight(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Check the remote host over ssh before syncing or training."""
    remote = _impl("msft.remote")
    host = _remote_host()
    display = getattr(remote, "mask_host", redact)(host)
    checks = _jsonable(_func(remote, "preflight")(host, execute=bool(args.execute)))
    payload = {"host": display, "checks": checks}
    text = "\n".join(
        f"{row.get('status', '?'):<4} {row.get('name')}: {row.get('detail')}" for row in checks
    )
    _emit(args.as_json, payload, f"remote {display}:\n{text}")
    return EXIT_OK


def _cmd_remote_sync(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Copy the dataset or a directory to the remote host."""
    remote_mod = _impl("msft.remote")
    host = _remote_host()
    local = args.local
    remote_path = args.remote_path
    if local is None:
        local = _require_config(cfg).data.input
    if remote_path is None:
        remote_path = _require_config(cfg).output_dir
    result = _jsonable(
        _func(remote_mod, "sync")(host, Path(local), str(remote_path), execute=bool(args.execute))
    )
    _emit(args.as_json, result, _render("remote sync", result))
    return EXIT_OK


def _cmd_remote_run(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Run the whole pipeline on the remote host."""
    config = _require_config(cfg)
    remote = _impl("msft.remote")
    host = _remote_host()
    result = _jsonable(_func(remote, "run_remote")(config, host, execute=bool(args.execute)))
    _emit(args.as_json, result, _render("remote run", result))
    return EXIT_OK


def _cmd_config_show(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Print the effective config (redacted) or its JSON form."""
    config = _require_config(cfg)
    for warning in validate_config(config):
        _log(f"warning: {warning}")
    if args.as_json:
        _emit(True, config.to_dict(), "")
        return EXIT_OK
    print(dump_config(config).rstrip("\n"))
    return EXIT_OK


# --- parser ---------------------------------------------------------------


def _common_flags() -> argparse.ArgumentParser:
    """Global flags, shared by the root parser and every subparser (both positions work)."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help=f"run config YAML (default: {DEFAULT_CONFIG.as_posix()} when it exists)",
    )
    common.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="print exactly one JSON object on stdout",
    )
    common.add_argument(
        "--execute",
        action="store_true",
        default=argparse.SUPPRESS,
        help="actually run the step; the default only prints the plan",
    )
    common.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="explicitly ask for the plan only (the default)",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree; every leaf sets a `(cfg, args) -> int` handler."""
    common = _common_flags()
    parser = argparse.ArgumentParser(
        prog="msft",
        description="Fine-tune the machinum-pipeline book model. Dry-run by default.",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"msft {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p_doctor = sub.add_parser(
        "doctor", parents=[common], help="check env, GPU, VRAM, disk and tokenizer"
    )
    p_doctor.set_defaults(handler=_cmd_doctor)

    p_data = sub.add_parser("data", parents=[common], help="dataset mining and splitting")
    data_sub = p_data.add_subparsers(dest="data_command", metavar="ACTION", required=True)
    p_data_build = data_sub.add_parser(
        "build", parents=[common], help="build chat triples from pipeline artifacts"
    )
    p_data_build.set_defaults(handler=_cmd_data_build)
    p_data.set_defaults(handler=_cmd_data_build)

    p_train = sub.add_parser("train", parents=[common], help="QLoRA/SFT via Unsloth + TRL")
    p_train.add_argument("--script", action="store_true", help="print the generated script")
    p_train.set_defaults(handler=_cmd_train)

    p_merge = sub.add_parser("merge", parents=[common], help="merge the adapter into 16-bit")
    p_merge.add_argument("--script", action="store_true", help="print the generated script")
    p_merge.set_defaults(handler=_cmd_merge)

    p_quant = sub.add_parser("quantize", parents=[common], help="GGUF conversion and quantisation")
    p_quant.add_argument("--script", action="store_true", help="print the generated script")
    p_quant.set_defaults(handler=_cmd_quantize)

    p_eval = sub.add_parser("eval", parents=[common], help="protocol gates on predictions")
    p_eval.add_argument(
        "--predictions",
        metavar="PATH",
        default=None,
        help="predictions JSONL (default: <output_dir>/eval/predictions.jsonl)",
    )
    p_eval.set_defaults(handler=_cmd_eval)

    p_remote = sub.add_parser("remote", parents=[common], help="ssh orchestration on a GPU box")
    remote_sub = p_remote.add_subparsers(dest="remote_command", metavar="ACTION", required=True)
    p_preflight = remote_sub.add_parser(
        "preflight", parents=[common], help="check the remote host (host from MSFT_REMOTE_HOST)"
    )
    p_preflight.set_defaults(handler=_cmd_remote_preflight)
    p_sync = remote_sub.add_parser("sync", parents=[common], help="copy data to the remote host")
    p_sync.add_argument("--local", metavar="PATH", default=None, help="local path to send")
    p_sync.add_argument(
        "--remote", dest="remote_path", metavar="PATH", default=None, help="destination directory"
    )
    p_sync.set_defaults(handler=_cmd_remote_sync)
    p_run = remote_sub.add_parser(
        "run", parents=[common], help="run the configured pipeline remotely"
    )
    p_run.set_defaults(handler=_cmd_remote_run)

    p_config = sub.add_parser("config", parents=[common], help="inspect the effective config")
    config_sub = p_config.add_subparsers(dest="config_command", metavar="ACTION", required=True)
    p_show = config_sub.add_parser("show", parents=[common], help="print the resolved config")
    p_show.set_defaults(handler=_cmd_config_show)

    return parser


# --- entry point ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse `argv`, run the selected handler and map failures to exit codes."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help / --version / bad usage
        return int(exc.code) if exc.code else EXIT_OK

    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - argparse requires a subcommand
        parser.print_help(sys.stderr)
        return EXIT_ERROR

    args.as_json = bool(getattr(args, "as_json", False))
    args.execute = bool(getattr(args, "execute", False))
    args.dry_run = bool(getattr(args, "dry_run", False))
    args.config = getattr(args, "config", None)

    if args.dry_run and args.execute:
        _log("error: --dry-run and --execute are mutually exclusive")
        return EXIT_ERROR

    try:
        return int(handler(_resolve_config(args), args))
    except MsftError as exc:
        _log(f"error: {exc}")
        return EXIT_ERROR
    except _NotImplementedYet as exc:
        payload = {"status": "not_implemented", "module": exc.module, "error": exc.message}
        if args.as_json:
            print(json.dumps(payload))
        else:
            _log(f"error: {exc.message}")
        return EXIT_NOT_IMPLEMENTED
    except KeyboardInterrupt:
        _log("error: interrupted")
        return 130
