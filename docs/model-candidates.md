# Candidate models, checked against the registry

A list of five models circulated on Twitter as "the best open models for a local box". This page
records what each name actually resolves to, because four of the five are *families*, not
artefacts, and the abliterated versions are almost all third-party re-uploads.

Method: every entry was checked against the ollama registry (official `library/` tags and the
community search), and the three models already on the GPU box were asked for their real
parameter counts with `POST /api/show`. No benchmark number is quoted from a screenshot.

## What the list resolves to

| claim | real artefact | size (Q4) | on the box | fits one 16 GB card |
| --- | --- | --- | --- | --- |
| **Qwen3.8-27B** "overall" | `library/qwen3.8:27b` — verified **27.3 B**, 262 144 ctx, Q4_K_M | 17 GB on disk, **20 GB resident at 32 k ctx** | **yes** (`qwen3.8:latest`) | no — needs both cards (measured at 100 % GPU with 2×16 GB) |
| **Gemma 4 12B** "small GPU" | `library/gemma4:12b-it-q4_K_M` | ≈ 8 GB | no — the box has `gemma4:latest` = **7.5 B** and `gemma4:31b` = **30.7 B** | yes, comfortable |
| **Gemma 4 26B-A4B** "fastest" | `library/gemma4:26b-a4b-it-q4_K_M` — MoE, ~4 B active per token | ≈ 15–16 GB | no | yes, barely; decode speed of a 4 B model |
| **Qwen3.6-35B-A3B** "agentic" | `library/qwen3.6:35b-a3b` — MoE, 3 B active | ≈ 21 GB | no | no — both cards |
| **Mistral Small 3.2** "writing" | `library/mistral-small3.2:24b` | ≈ 14 GB | no | yes |

Abliterated / uncensored versions exist for most of them, but they are community re-uploads:

| family | abliterated artefact | note |
| --- | --- | --- |
| Qwen3.8 | `huihui_ai/Qwen3.8-abliterated`, `kiwi_kiwi/Qwen3.8-abliterated-27b` | huihui_ai is the long-running abliteration publisher; the kiwi_kiwi tag is a re-quant |
| Qwen3.6 | `huihui_ai/Qwen3.6-abliterated`; 35B-A3B uncensored via `satgeze/qwen36-35b-uncensored-1m`, `eriktonon/qwen3.6-moe-uncensored` | the 35B variants are one-person builds with custom templates |
| Gemma４ | no `huihui_ai` build found; `bryonahigbee/gemma4-abliterated-DevDocs`, `mgraffam/gemma4-agentic-fable-abliterated`, `eriktonon/gemma4-uncensored` | hobby uploads; no upstream benchmark |
| Mistral Small | `huihui_ai/mistral-small-abliterated` | 24 B, ≈ 14 GB |

## Does any of it fit our setup?

Our constraint is narrower than "runs on a 16 GB card". The job is: one line of English prose in,
one line of Russian out, thousands of times per book, over the OpenAI-compatible endpoint that
`mt` speaks, with ≥32 K context, from a GGUF, on a box that must also be free to train.

1. **Abliteration is the wrong axis.** It removes refusal behaviour. Literary translation never
   triggers a refusal; what it needs is Russian morphology, register and consistency. Abliterated
   weights are produced by third parties without published evals, and each one is a new
   supply-chain and template risk — the same class of risk that cost us two runs (a hand-copied
   Gemma template in front of a Qwen model). Use the base model.
2. **Speed is not the bottleneck.** The pipeline is line-serial with concurrency 4; a 4 B model
   already answers a line faster than the pipeline can schedule it. A "fastest" MoE buys nothing
   here.
3. **The real bottleneck is Russian quality**, and that scales with parameters up to a point:
   the 4 B we shipped drifts names, produces non-words ("в багажаре") and calques. A 26–27 B model
   at Q4 is the smallest size that showed none of those on the same chapter.
4. **VRAM is the budget that matters**, because the same cards train the next adapter.

Verdict for this repo:

| role | model | why |
| --- | --- | --- |
| keep as the fast/cheap artefact | **`machinum:4b`** (Qwen3.5-4B + LoRA) | 2.6 GB, 100 % GPU, learns the protocol; fine for smoke tests and gate runs |
| best quality that still leaves room | **`gemma4:26b-a4b-it-q4_K_M`** | ≈ 16 GB, MoE so decode cost is 4 B, 26 B of knowledge; still one card |
| best quality, training must wait | **`qwen3.8:27b`** (already installed) | 20 GB resident at 32 k, 262 k ctx, strongest Russian of the three we measured |
| not recommended | any abliterated re-upload, Mistral Small 3.2 | unvetted weights; Mistral is a 2025 model and behind Qwen/Gemma on Russian |

## What to do next, in order

1. Run the same held-out chapter through `gemma4:26b-a4b-it-q4_K_M` and `qwen3.8:27b` and keep the
   diff — the 4 B baseline is now recorded, so the comparison is one command.
2. Only consider fine-tuning a 26–27 B model after the dataset is a few thousand pairs; a LoRA at
   that size needs QLoRA on 16 GB and no concurrent training job.
3. Never ship an abliterated model as the pipeline's default. If one is tried, it goes through
   `msft eval` like everything else.
