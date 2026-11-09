# Module contract (read before editing)

Five modules are built in parallel. **Only touch the files assigned to you.** If you need a
change in someone else's file, write it in your report instead of editing.

## Files and owners

| Path | Contents |
| --- | --- |
| `src/msft/__init__.py`, `__main__.py` | package metadata, `python -m msft` entry |
| `src/msft/cli.py`, `src/msft/config.py`, `tests/test_cli.py`, `tests/test_config.py` | argument parsing, subcommand dispatch, config dataclasses/validation/redaction |
| `src/msft/doctor.py`, `src/msft/remote.py`, `tests/test_doctor.py`, `tests/test_remote.py` | environment probing, ssh orchestration |
| `src/msft/data.py`, `tests/test_data.py`, `tests/fixtures/*.jsonl` | dataset mining, filtering, splitting, stats |
| `src/msft/train.py`, `src/msft/merge.py`, `src/msft/quantize.py`, `tests/test_train.py`, `tests/test_quantize.py` | training plan/script/execution, adapter merge, GGUF |
| `src/msft/evaluate.py`, `tests/test_evaluate.py`, `configs/*.yaml`, `.github/workflows/ci.yml` | protocol gates, ready-made configs, CI |

## Hard rules

1. **Python ≥3.10**, `from __future__ import annotations` at the top of every module.
2. **Runtime deps in core modules: stdlib + PyYAML only.** Import `torch`/`unsloth`/`trl`
   lazily *inside* functions, never at module import time, so `msft --help` works without them.
3. **No secrets, no hosts, no usernames, no IPs** in code, tests, configs or docs. Remote host
   comes from `MSFT_REMOTE_HOST`; a report prints it redacted (`user@ho***`).
4. **Nothing that trains or downloads runs without `--execute`.** The default is a printed
   plan (`--dry-run` behaviour), exit code 0.
5. **Tests must pass offline and without a GPU**: `pytest -q`. No network, no model downloads.
   Use `tmp_path`, monkeypatch `subprocess.run`, and tiny 3–5 record fixtures.
6. **Style:** `ruff check .` clean, line length 100, type hints everywhere, one-line docstring
   per public function. Errors are raised as `MsftError` (defined in `config.py`) and the CLI
   catches them, printing `error: <msg>` to stderr and returning exit code 2. An unexpected
   exception is not a red gate: it is logged with a traceback, reported the same way and also
   returns 2, so a CI wrapper cannot confuse "the code crashed" with "the gates failed" (1).
   A command that ran and failed — a failed rsync, a non-zero training run — returns 1.
7. **Every command supports `--json`** and prints a single JSON object when it is set.

## Interfaces (do not change signatures without telling the others)

```python
# config.py
@dataclass class ModelCfg:  id: str; revision: str | None = None; load_in_4bit: bool = True; dtype: str = "bfloat16"
@dataclass class LoraCfg:   r: int = 32; alpha: int = 16; dropout: float = 0.0
                            target_modules: list[str] = field(default_factory=lambda: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
                            modules_to_save: list[str] = field(default_factory=lambda: ["lm_head"]); use_rslora: bool = False
@dataclass class TrainCfg:  max_seq_length: int = 4096; per_device_batch: int = 1; grad_accum: int = 32
                            lr: float = 5e-5; epochs: float = 2.0; warmup_steps: int = 8
                            weight_decay: float = 0.05; max_grad_norm: float = 1.0
                            scheduler: str = "cosine"; optim: str = "adamw_8bit"; seed: int = 42
                            packing: bool = False; completion_only_loss: bool = True
                            save_steps: int = 20; eval_steps: int = 20
@dataclass class DataCfg:   input: str = "data/raw"; val_ratio: float = 0.03; dedupe: bool = True
                            min_chars: int = 50; max_chars: int = 8000; max_tokens: int | None = None
@dataclass class QuantCfg:  method: str = "q4_k_m"; imatrix: bool = False; output: str = "out/model.gguf"
@dataclass class RunConfig: name: str; model: ModelCfg; lora: LoraCfg; train: TrainCfg; data: DataCfg
                            quantize: QuantCfg; output_dir: str = "out"; base_dir: str = "."
load_config(path: str | Path) -> RunConfig
validate_config(cfg: RunConfig) -> list[str]      # warnings; raises MsftError on fatal
redact(value: str) -> str                          # masks hosts/IPs/keys in any printed string
```

```python
# doctor.py
@dataclass class Check: name: str; status: str  # "ok" | "warn" | "fail"; detail: str
run_doctor(cfg: RunConfig | None = None) -> list[Check]
format_report(checks: list[Check], as_json: bool) -> str
```

```python
# data.py
@dataclass class DataStats: records_in: int; records_out: int; dropped: dict[str, int]
                            train: int; val: int; chars: int; approx_tokens: int; sources: dict[str, int]
build_dataset(cfg: RunConfig, *, execute: bool = False) -> DataStats
iter_records(path: Path) -> Iterator[dict]         # accepts .jsonl and .json arrays
normalize_record(raw: dict) -> dict | None          # -> {"messages": [...], "source": str}
```

```python
# train.py / merge.py / quantize.py
def plan(cfg: RunConfig) -> dict                    # JSON-serialisable plan
def build_script(cfg: RunConfig) -> str             # a standalone python script body
def run(cfg: RunConfig, *, execute: bool = False) -> dict
```

```python
# evaluate.py
@dataclass class GateResult: gate: str; passed: bool; detail: str; value: float | None = None
run_gates(predictions: Path, *, as_json: bool = False) -> list[GateResult]
```

```python
# remote.py
def preflight(host: str, *, execute: bool = False) -> list[Check]
def sync(host: str, local: Path, remote: str, *, execute: bool = False) -> dict
def run_remote(cfg: RunConfig, host: str, *, execute: bool = False) -> dict
```

## Definition of done for each module

- `python -m msft <cmd> --help` works with no heavy deps installed.
- `pytest -q tests/test_<yours>.py` passes.
- `ruff check src/msft/<yours>.py tests/test_<yours>.py` clean.
- `--json` output is a single object, sorted keys not required, no log lines on stdout.
