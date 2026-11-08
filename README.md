# machinum-sft

Fine-tuning CLI for the model that processes books in
[`machinum-pipeline`](https://github.com/glebson/machinum-pipeline): translation EN→RU,
proofreading and text normalization against a strict line/XML protocol.

The point of this repo is a **repeatable, boring** run: dataset in, adapter out, GGUF out,
gates green. Everything that touches hardware, paths or secrets is configuration; nothing is
hardcoded.

```bash
uv venv --python 3.11 && . .venv/bin/activate
uv pip install -e ".[dev]"
python -m msft --help
python -m msft doctor
```

## What it does

| Command | What it is for |
| --- | --- |
| `msft doctor` | env/GPU/VRAM/disk/tokenizer check before you burn hours |
| `msft data build` | mine the pipeline's JSONL artifacts into chat triples, filter, dedupe, split |
| `msft train` | QLoRA/SFT via Unsloth + TRL; `--dry-run` prints the exact plan and script |
| `msft merge` | merge the adapter into 16-bit weights |
| `msft quantize` | GGUF conversion, optional imatrix calibration, quantise to Q4_K_S/Q5_0 |
| `msft eval` | protocol gates: parse rate, line parity, schema validity, chatter rate |
| `msft remote` | run the same config on a remote GPU box over ssh (host comes from `.env`) |

## Design rules

- **Core modules are stdlib-only** (+ PyYAML). Heavy deps (`torch`, `unsloth`, `trl`) live in
  the `train` extra and are only imported inside the code paths that need them.
- **`--dry-run` is the default for anything that trains or downloads.** You opt into work.
- **No secrets, no hosts in the tree.** `.env` is git-ignored; `.env.example` shows the keys.
- **Tests never touch the network or a GPU.**

See `CONTRACT.md` for the module interfaces, and `docs/model-choice.md` for the base-model
decision this repo follows.

MIT licensed.
