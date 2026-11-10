# Model tags: what each one is, and which one to run

The box serves one ollama instance; the machinum family is several tags of the same base
(Qwen3.5-4B) with different provenance. A tag is only meaningful next to the corpus it learned
from and the surgery it was given, so this page records both.

| tag | id | size | what it is | corpus | how it was made |
| --- | --- | --- | --- | --- | --- |
| `machinum:4b` | `3337790d35d5` | 2.6 GB | the shipped artefact: protocol learned, language is the base model's | 49 single-line pairs | `configs/qwen3.5-4b-lora.yaml` -> merge -> Q4_K_S |
| `machinum:4b-v2-data` | `7a379d95227a` | 2.6 GB | second run, data arm: same hyperparameters, 2.7x corpus with multi-line samples | 135 records (49 single + 86 grouped) | `configs/machinum-4b-v2-data.yaml` |
| `machinum:4b-v2-tuned` | `f97436212f97` | 2.6 GB | second run, tuned arm: r=16, alpha=32, lr 1e-4 on the same corpus | same | `configs/machinum-4b-v2-tuned.yaml` |
| `machinum:4b-heretic` | `204984f5f7d1` | 2.6 GB | `machinum:4b` with the refusal direction projected out | same as `machinum:4b` | `scripts/abliterate.py --execute`, then Q4_K_S |

`machinum:4b-fullseq` carries the same digest as `machinum:4b` — two tags, one set of weights.
`machinum:4b-cov2` is an earlier experiment and is not part of this comparison.

## Which to run

- **Pipeline / production:** `machinum:4b-v2-data`. More data, unchanged recipe, best BLEU of the
  four (45.6 with the training prompt, 45.4 with the default one) and no regression in the gates.
- **Uncensored prose:** `machinum:4b-heretic`. Same translation behaviour, no refusal on content
  the base model declines.
- **Reproducing an old number:** `machinum:4b`, then quote `docs/index.md` alongside it.

## What the comparison actually showed

Six EN→RU cases, `gpt-oss:20b` judge, `--seed 7 --num-predict 4096`, default arena prompt:

| model | schema | chrF | BLEU | judge f/f |
| --- | --- | --- | --- | --- |
| `machinum:4b-heretic` | 0.833 | 62.7 | 45.4 | 3.67/3.50 |
| `machinum:4b-v2-tuned` | 0.833 | 61.0 | 41.5 | 4.33/4.00 |
| `machinum:4b-v2-data` | 0.667 | 63.3 | 43.5 | 4.33/4.33 |
| `machinum:4b` | 0.667 | 63.1 | 43.3 | 4.17/4.00 |

Same cases, same models, the corpus builder's own system prompt instead:

| model | schema | chrF | BLEU |
| --- | --- | --- | --- |
| `machinum:4b-v2-data` | **1.000** | 62.9 | **45.6** |
| `machinum:4b` | **1.000** | 61.9 | 42.5 |
| `machinum:4b-v2-tuned` | **1.000** | 60.5 | 43.2 |

Two conclusions, in order of importance:

1. **Line parity follows the prompt, not (only) the corpus.** Under the prompt the model was
   trained with, every tag answers multi-line input line for line; under the arena's default
   prompt the multi-line cases collapse again. The capability is in the weights, but the
   instruction that switches it on is part of the model's contract.
2. **Data beat the hyperparameter arm.** `v2-data` gains ~3 BLEU over `machinum:4b` with the same
   recipe; `v2-tuned` (r=16/alpha=32, lr 1e-4) gains nothing on chrF and slightly loses BLEU.
   On 135 records, the corpus is the constraint, not the rank.

Abliteration is not free: `machinum:4b-heretic` drops the judge's fidelity/fluency by 0.5/0.5
against its parent, and gains 2 BLEU. It is a content-policy trade, not a quality upgrade — use it
where refusals matter and keep `v2-data` for the pipeline.

## The next thing to fix

The product does not send either of those prompts. It sends
`packages/tools/prompts/translate.system.md` — literary guidelines, "output exactly one translated
line per input line; do not add, merge or drop lines", an optional glossary block and a trailing
`<terms>` block. The adapter has never seen that instruction, so the honest third run renders that
prompt (with the placeholders filled for English -> Russian and no glossary) into the corpus, and
`msft arena` grows a `--system-file` so the comparison uses it too.

## Reproducing a tag

```bash
# data -> training -> quantise, on the GPU box through the queue
python scripts/pairs_from_book.py --en <book dir> --out data/v2/corpus.jsonl --groups 2,3
msft data build --config configs/machinum-4b-v2-data.yaml --execute
msft remote sync --local data/v2 --execute
msft remote signal data     --config configs/machinum-4b-v2-data.yaml --id 10-data  --timeout 20m --execute
msft remote signal train    --config configs/machinum-4b-v2-data.yaml --id 20-train --timeout 40m --execute
msft remote signal merge    --config configs/machinum-4b-v2-data.yaml --id 30-merge --timeout 20m --execute
msft remote signal quantize --config configs/machinum-4b-v2-data.yaml --id 40-quant --timeout 30m --execute

# the heretic twin
python scripts/abliterate.py --model out/merged-qwen3.5-4b-lora --out out/abliterated-machinum-4b --execute
python /opt/llama.cpp/convert_hf_to_gguf.py out/abliterated-machinum-4b \
    --outfile out/model-machinum-4b-heretic-f16.gguf --outtype f16
/opt/llama.cpp/build/bin/llama-quantize out/model-machinum-4b-heretic-f16.gguf \
    out/model-machinum-4b-heretic.gguf Q4_K_S
OLLAMA_CONTAINER=ollama bash scripts/ollama-export.sh out/model-machinum-4b-heretic.gguf machinum:4b-heretic
```

Queue ids sort lexicographically by (mtime, name), so number them `10-`, `20-`, `30-` — a job that
must follow another one cannot rely on the order it was typed in.
