"""Command-line parsing and dispatch for `msft`.

Only argparse is used here, and every implementation module is imported *inside* its handler,
so `python -m msft --help` stays usable on a machine without torch, unsloth or trl.

Exit codes: 0 success, 1 red gates or a command that failed, 2 user/config error or an
internal error that prevented the run (`error: ...` on stderr), 3 the requested implementation
module is not written yet. Every step that would train or download is a printed plan unless
`--execute` is passed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
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


def _exit_for(result: dict[str, Any]) -> int:
    """Exit code for a result mapping: a plan is ok, an executed step follows ok/returncode."""
    if result.get("executed") is False:
        return EXIT_OK
    if "ok" in result:
        return EXIT_OK if result["ok"] else 1
    return EXIT_OK if result.get("returncode", 0) == 0 else 1


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


def _model_stage(
    module: str,
    title: str,
    cfg: RunConfig | None,
    args: argparse.Namespace,
    **run_kwargs: Any,
) -> int:
    """Shared body for train/merge/quantize: script, dry-run plan, or execution."""
    config = _require_config(cfg)
    stage = _impl(module)
    if getattr(args, "script", False):
        script = _func(stage, "build_script")(config)
        _emit(args.as_json, {"script": script}, str(script))
        return EXIT_OK
    if args.execute:
        result = _jsonable(_func(stage, "run")(config, execute=True, **run_kwargs))
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
    inspect = bool(getattr(args, "inspect_mask", False))
    if inspect and args.dry_run:
        raise MsftError(
            "--inspect-mask runs the training script; it cannot be combined with --dry-run"
        )
    if inspect:
        # Inspection loads the model and renders one batch; it is an explicit run request.
        args.execute = True
    return _model_stage("msft.train", "train", cfg, args, inspect_mask=inspect)


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


def _cmd_fetch(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Download the base model with resume and retries (config or --model)."""
    fetch = _impl("msft.fetch")
    model = args.model
    revision = None
    if not model:
        config = _require_config(cfg)
        model = config.model.id
        revision = config.model.revision
    result = _jsonable(fetch.run(model, revision=revision, attempts=args.attempts))
    _emit(args.as_json, result, _func(fetch, "format_report")(result))
    return EXIT_OK if result.get("ok") else 1


def _cmd_verify(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Check the merged model, its tokenizer and the quantised GGUF."""
    config = _require_config(cfg)
    verify = _impl("msft.verify")
    result = _jsonable(_func(verify, "run")(config))
    _emit(args.as_json, result, _func(verify, "format_report")(result))
    return EXIT_OK if result.get("ok") else 1


def _cmd_export(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Create (and optionally push) an ollama model from a GGUF."""
    export = _impl("msft.export")
    # Only an explicit --config asks for the verify gate; a default config picked up from the
    # cwd must not silently change what `export` does.
    gate = cfg if getattr(args, "config", None) else None
    result = _jsonable(
        _func(export, "run")(
            args.gguf,
            args.name,
            namespace=args.namespace,
            push=bool(args.push),
            container=args.container,
            config=gate,
        )
    )
    _emit(args.as_json, result, _func(export, "format_report")(result))
    return EXIT_OK if result.get("ok") else 1


def _cmd_probe(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Chat through the serve path with think:false, then run the eval gates."""
    probe = _impl("msft.probe")
    system = args.system or str(getattr(probe, "SYSTEM_PROMPT", ""))
    result = _jsonable(
        _func(probe, "run")(args.endpoint, args.model, args.cases, system=system)
    )
    _emit(args.as_json, result, _func(probe, "format_report")(result))
    return EXIT_OK if result.get("ok") else 1


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
    return 1 if any(row.get("status") == "fail" for row in checks) else EXIT_OK


def _cmd_remote_sync(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Copy a directory under `data/` to `<remote repo>/data/` on the box."""
    remote_mod = _impl("msft.remote")
    if getattr(args, "remote_path", None):
        raise MsftError("--remote was removed: the destination is always <remote repo>/data/…")
    host = _remote_host()
    local = args.local
    if local is None:
        local = cfg.data.input if cfg is not None else "data"
    base_dir = Path(cfg.base_dir) if cfg is not None else Path(".")
    result = _jsonable(
        _func(remote_mod, "sync")(
            host,
            local,
            base_dir=base_dir,
            delete=bool(args.delete),
            execute=bool(args.execute),
        )
    )
    _emit(args.as_json, result, _render("remote sync", result))
    return _exit_for(result)


def _parse_timeout(value: str | None) -> int | None:
    """Parse `--timeout DURATION` into seconds; absent means no limit."""
    if value is None:
        return None
    worker = _impl("msft.worker")
    return int(_func(worker, "parse_duration")(value))


def _cmd_remote_run(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Run the whole pipeline on the remote host."""
    config = _require_config(cfg)
    remote = _impl("msft.remote")
    host = _remote_host()
    result = _jsonable(
        _func(remote, "run_remote")(
            config,
            host,
            execute=bool(args.execute),
            timeout_s=_parse_timeout(args.timeout),
        )
    )
    _emit(args.as_json, result, _render("remote run", result))
    return _exit_for(result)


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


def _cmd_remote_deploy(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Pull the pushed branch on the box, then rebuild or restart the worker container."""
    deploy = _impl("msft.deploy")
    spec = deploy.load_spec()
    restart = bool(getattr(args, "restart", False))
    if restart and args.no_build:
        raise MsftError("--restart and --no-build are mutually exclusive")
    mode = "restart" if restart else ("up --no-build" if args.no_build else "up --build")
    pulled = _jsonable(deploy.pull(spec, execute=bool(args.execute)))
    if pulled.get("executed") and _exit_for(pulled) != EXIT_OK:
        payload = {
            "mode": mode,
            "spec": deploy.spec_json(spec),
            "pull": pulled,
            "up": None,
            "error": "pull failed; not rebuilding on stale code",
        }
        text = "\n".join(
            [
                f"remote {deploy.spec_json(spec)['host']}  {spec.repo}",
                f"pull    rc={pulled.get('returncode')}",
                f"{mode:<7} skipped: the pull failed, so the box was not rebuilt",
            ]
        )
        _emit(args.as_json, payload, text)
        return 1
    if restart:
        action = _jsonable(deploy.restart(spec, execute=bool(args.execute)))
    else:
        action = _jsonable(deploy.up(spec, execute=bool(args.execute), build=not args.no_build))
    payload = {"mode": mode, "spec": deploy.spec_json(spec), "pull": pulled, "up": action}
    pull_line = "rc=" + str(pulled.get("returncode")) if pulled.get("executed") else "plan"
    action_line = "rc=" + str(action.get("returncode")) if action.get("executed") else "plan"
    text = "\n".join(
        [
            f"remote {deploy.spec_json(spec)['host']}  {spec.repo}",
            f"pull    {pull_line}",
            f"{mode:<7} {action_line}",
        ]
    )
    _emit(args.as_json, payload, text)
    return _exit_for(action)


def _cmd_remote_down(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Stop the worker container on the box."""
    deploy = _impl("msft.deploy")
    spec = deploy.load_spec()
    result = _jsonable(deploy.down(spec, execute=bool(args.execute)))
    _emit(args.as_json, result, f"down rc={result.get('returncode', 'plan')}")
    return _exit_for(result)


def _cmd_remote_signal(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Queue a job on the box through the running container."""
    deploy = _impl("msft.deploy")
    spec = deploy.load_spec()
    result = _jsonable(
        deploy.signal(
            spec,
            args.kind,
            config=_config_arg(cfg, args),
            job_id=args.job_id,
            extra_args=args.extra_args,
            timeout_s=_parse_timeout(args.timeout),
            execute=bool(args.execute),
        )
    )
    _emit(args.as_json, result, f"signal {args.kind} -> rc={result.get('returncode', 'plan')}")
    return _exit_for(result)


def _cmd_remote_status(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Container health and the worker queue from the box."""
    deploy = _impl("msft.deploy")
    spec = deploy.load_spec()
    result = _jsonable(deploy.status(spec, execute=bool(args.execute)))
    _emit(args.as_json, result, result.get("stdout") or "plan")
    return _exit_for(result)


def _worker_dir(cfg: RunConfig | None, explicit: str | None) -> str:
    """Worker directory: --dir, else MSFT_WORKER_DIR, else <repo>/.worker."""
    if explicit:
        return explicit
    env = os.environ.get("MSFT_WORKER_DIR")
    if env:
        return env
    if cfg is not None:
        return str(Path(cfg.base_dir) / ".worker")
    return ".worker"


def _config_arg(cfg: RunConfig | None, args: argparse.Namespace) -> str | None:
    """The config path to hand to the worker (the file the user actually passed)."""
    explicit = getattr(args, "config", None)
    if explicit:
        return str(explicit)
    if cfg is not None:
        return str(Path(cfg.base_dir) / "configs" / f"{cfg.name}.yaml")
    return None


def _cmd_worker_serve(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Serve the job queue: one job at a time, forever (or once)."""
    worker = _impl("msft.worker")
    directory = _worker_dir(cfg, args.dir)
    if args.as_json:
        _emit(True, {"dir": directory, "interval": args.interval, "once": bool(args.once)}, "")
        return EXIT_OK
    return int(worker.serve(directory, interval=args.interval, once=bool(args.once)))


def _cmd_worker_signal(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Queue a job for the worker."""
    worker = _impl("msft.worker")
    directory = _worker_dir(cfg, args.dir)
    timeout_s = _parse_timeout(args.timeout)
    job_id = worker.signal(
        directory,
        args.kind,
        job_id=args.job_id,
        config=_config_arg(cfg, args),
        extra_args=args.extra_args,
        timeout_s=timeout_s,
    )
    timeout_note = "none" if timeout_s is None else f"{timeout_s}s"
    payload = {"id": job_id, "dir": directory, "timeout": timeout_note}
    _emit(
        args.as_json,
        payload,
        f"queued {job_id} in {directory} (timeout: {timeout_note})",
    )
    return EXIT_OK


def _cmd_worker_status(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Print the queue state and the heartbeat age."""
    worker = _impl("msft.worker")
    directory = _worker_dir(cfg, args.dir)
    state = _jsonable(worker.status(directory))
    heartbeat = state.get("heartbeat") or {}
    counts = state.get("counts") or {}
    age = heartbeat.get("age_s")
    lines = [
        f"worker  {state.get('dir')}",
        f"{state.get('gpu', 'GPU: unknown')}",
        f"state   {heartbeat.get('state', 'no heartbeat')}"
        + (f" (updated {age}s ago)" if isinstance(age, (int, float)) else ""),
        f"queue   pending={counts.get('pending', 0)} running={counts.get('running', 0)}"
        f" done={counts.get('done', 0)} failed={counts.get('failed', 0)}",
    ]
    for job in state.get("running") or []:
        lines.append(f"running {job.get('id')} {job.get('kind')} since {job.get('started')}")
    for job in (state.get("done") or [])[-3:]:
        lines.append(
            f"{job.get('status', '?'):<6} {job.get('id')} rc={job.get('returncode')}"
            f" {job.get('duration_s')}s"
        )
    _emit(args.as_json, state, "\n".join(lines))
    return EXIT_OK


def _cmd_worker_log(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Tail one job log."""
    worker = _impl("msft.worker")
    directory = _worker_dir(cfg, args.dir)
    text = worker.tail_log(directory, args.job_id, lines=args.lines)
    if args.as_json:
        _emit(True, {"dir": directory, "job": args.job_id, "tail": text}, text)
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return EXIT_OK


def _cmd_worker_reset(cfg: RunConfig | None, args: argparse.Namespace) -> int:
    """Clear inbox/running/done, keeping the logs."""
    worker = _impl("msft.worker")
    directory = _worker_dir(cfg, args.dir)
    removed = worker.reset(directory)
    _emit(args.as_json, {"dir": directory, "removed": removed}, f"cleared {removed} entries")
    return EXIT_OK


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
    p_train.add_argument(
        "--inspect-mask",
        action="store_true",
        help="run the script with MSFT_INSPECT_MASK=1 to report the loss mask, then stop",
    )
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

    p_fetch = sub.add_parser(
        "fetch", parents=[common], help="resumable HF weight download (retries, checks incomplete)"
    )
    p_fetch.add_argument(
        "--model",
        metavar="HF_ID",
        default=None,
        help="model id to download (default: the config's model.id)",
    )
    p_fetch.add_argument("--attempts", type=int, default=8, help="download attempts (default: 8)")
    p_fetch.set_defaults(handler=_cmd_fetch)

    p_verify = sub.add_parser(
        "verify", parents=[common], help="check merged config, tokenizer and GGUF size"
    )
    p_verify.set_defaults(handler=_cmd_verify)

    p_export = sub.add_parser(
        "export", parents=[common], help="create (and optionally push) an ollama model from a GGUF"
    )
    p_export.add_argument(
        "--gguf", required=True, metavar="PATH", help="quantised GGUF to export"
    )
    p_export.add_argument(
        "--name", required=True, help="ollama model name, e.g. machinum:4b"
    )
    p_export.add_argument(
        "--namespace", metavar="USER", default=None, help="ollama.com namespace for cp/push"
    )
    p_export.add_argument(
        "--push", action="store_true", help="push <namespace>/<name> after create"
    )
    p_export.add_argument(
        "--container",
        default="ollama",
        metavar="NAME",
        help="ollama container to copy into (default: ollama; pass '' for a host binary)",
    )
    p_export.set_defaults(handler=_cmd_export)

    p_probe = sub.add_parser(
        "probe", parents=[common], help="chat through the serve path, then run the eval gates"
    )
    p_probe.add_argument("--endpoint", required=True, metavar="URL", help="ollama base URL")
    p_probe.add_argument("--model", required=True, help="served model name")
    p_probe.add_argument("--cases", required=True, metavar="PATH", help="cases JSONL")
    p_probe.add_argument(
        "--system", metavar="TEXT", default=None, help="system prompt (default: translation)"
    )
    p_probe.set_defaults(handler=_cmd_probe)

    p_remote = sub.add_parser("remote", parents=[common], help="ssh orchestration on a GPU box")
    remote_sub = p_remote.add_subparsers(dest="remote_command", metavar="ACTION", required=True)
    p_preflight = remote_sub.add_parser(
        "preflight", parents=[common], help="check the remote host (host from MSFT_REMOTE_HOST)"
    )
    p_preflight.set_defaults(handler=_cmd_remote_preflight)
    p_sync = remote_sub.add_parser("sync", parents=[common], help="copy data to the remote host")
    p_sync.add_argument("--local", metavar="PATH", default=None, help="path under data/ to send")
    p_sync.add_argument(
        "--delete",
        action="store_true",
        help="delete files on the destination that are gone locally (--delete-after)",
    )
    p_sync.add_argument(
        "--remote", dest="remote_path", metavar="PATH", default=None, help=argparse.SUPPRESS
    )
    p_sync.set_defaults(handler=_cmd_remote_sync)
    p_run = remote_sub.add_parser(
        "run", parents=[common], help="run the configured pipeline remotely"
    )
    p_run.add_argument("--timeout", metavar="DURATION", default=None, help="e.g. 900, 15m, 2h")
    p_run.set_defaults(handler=_cmd_remote_run)
    p_deploy = remote_sub.add_parser(
        "deploy", parents=[common], help="pull on the box and bring the worker container up"
    )
    p_deploy.add_argument("--no-build", action="store_true", help="skip the image rebuild")
    p_deploy.add_argument(
        "--restart",
        action="store_true",
        help="pull, then restart the worker container (use after worker code changed)",
    )
    p_deploy.set_defaults(handler=_cmd_remote_deploy)
    p_down = remote_sub.add_parser("down", parents=[common], help="stop the worker container")
    p_down.set_defaults(handler=_cmd_remote_down)
    p_signal = remote_sub.add_parser(
        "signal",
        parents=[common],
        help="queue a job on the box (train|merge|quantize|eval|fetch|verify|data)",
    )
    p_signal.add_argument("kind")
    p_signal.add_argument("--id", dest="job_id", default=None)
    p_signal.add_argument(
        "--arg", dest="extra_args", action="append", default=[], help="extra CLI arg (repeatable)"
    )
    p_signal.add_argument(
        "--timeout", metavar="DURATION", default=None, help="kill the job after e.g. 900, 15m, 2h"
    )
    p_signal.set_defaults(handler=_cmd_remote_signal)
    p_rstatus = remote_sub.add_parser(
        "status", parents=[common], help="container health plus the worker queue"
    )
    p_rstatus.set_defaults(handler=_cmd_remote_status)

    p_config = sub.add_parser("config", parents=[common], help="inspect the effective config")
    config_sub = p_config.add_subparsers(dest="config_command", metavar="ACTION", required=True)
    p_show = config_sub.add_parser("show", parents=[common], help="print the resolved config")
    p_show.set_defaults(handler=_cmd_config_show)

    p_worker = sub.add_parser(
        "worker", parents=[common], help="job queue served on the GPU box (signal / serve / status)"
    )
    worker_sub = p_worker.add_subparsers(dest="worker_command", metavar="ACTION", required=True)
    p_w_serve = worker_sub.add_parser("serve", parents=[common], help="serve the job queue")
    p_w_serve.add_argument("--dir", metavar="PATH", default=None, help="worker directory")
    p_w_serve.add_argument("--interval", type=float, default=15.0, help="poll interval, seconds")
    p_w_serve.add_argument("--once", action="store_true", help="run one job and exit")
    p_w_serve.set_defaults(handler=_cmd_worker_serve)
    p_w_signal = worker_sub.add_parser("signal", parents=[common], help="queue a job")
    p_w_signal.add_argument(
        "kind", help="train | merge | quantize | eval | fetch | verify | data | doctor"
    )
    p_w_signal.add_argument("--dir", metavar="PATH", default=None, help="worker directory")
    p_w_signal.add_argument(
        "--id", dest="job_id", default=None, help="job id (default: time-based)"
    )
    p_w_signal.add_argument(
        "--arg", dest="extra_args", action="append", default=[], help="extra CLI arg (repeatable)"
    )
    p_w_signal.add_argument(
        "--timeout", metavar="DURATION", default=None, help="kill the job after e.g. 900, 15m, 2h"
    )
    p_w_signal.set_defaults(handler=_cmd_worker_signal)
    p_w_status = worker_sub.add_parser("status", parents=[common], help="queue and heartbeat")
    p_w_status.add_argument("--dir", metavar="PATH", default=None, help="worker directory")
    p_w_status.set_defaults(handler=_cmd_worker_status)
    p_w_log = worker_sub.add_parser("log", parents=[common], help="tail a job log")
    p_w_log.add_argument("--dir", metavar="PATH", default=None, help="worker directory")
    p_w_log.add_argument("--id", dest="job_id", default=None, help="job id (default: latest)")
    p_w_log.add_argument("--lines", type=int, default=40)
    p_w_log.set_defaults(handler=_cmd_worker_log)
    p_w_reset = worker_sub.add_parser("reset", parents=[common], help="clear the queue, keep logs")
    p_w_reset.add_argument("--dir", metavar="PATH", default=None, help="worker directory")
    p_w_reset.set_defaults(handler=_cmd_worker_reset)

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
        if args.as_json:
            _emit(True, {"ok": False, "error": redact(str(exc))}, "")
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
        if args.as_json:
            _emit(True, {"ok": False, "error": "interrupted"}, "")
        return 130
    except Exception as exc:  # noqa: BLE001 - never let a traceback be the only output
        # An internal failure means "cannot run", not "the gates are red": exit 2, and keep
        # the --json contract with one object on stdout.
        message = redact(f"internal {type(exc).__name__}: {exc}")
        _log(message)
        print(redact(traceback.format_exc()), file=sys.stderr)
        if args.as_json:
            _emit(True, {"ok": False, "error": message}, "")
        return EXIT_ERROR
