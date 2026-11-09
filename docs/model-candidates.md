# Models this project uses

Two, and only two. Everything else that was evaluated is listed at the end with the measurement
that eliminated it — kept as evidence, not as a recommendation.

| role | model | why this one |
| --- | --- | --- |
| **fast / cheap** | `machinum:4b` — our Qwen3.5-4B + LoRA | 2.63 GB Q4_K_S, 3.2 GB resident at 8 k, 40 s per chapter, learns the protocol; the artefact this repo builds |
| **quality** | `igorls/gemma-4-12B-it-heretic-GGUF` | 7.4 GB Q4_K_M, 8.4 GB resident at 32 k, 60 s per chapter — the largest model that still leaves a 16 GB card free for training |

Both speak the OpenAI-compatible endpoint `mt` uses, both answer with `reasoning_effort: "none"`
(Gemma 4 declares `thinking`, so this is mandatory, not optional), both fit one card.

## How the two compare

Same held-out chapter (book2 ch.1), same filled glossary, same flags, one card, through `mt`:

| model | quant | size | resident @ 32k | chapter | prose verdict |
| --- | --- | --- | --- | --- | --- |
| `machinum:4b` | Q4_K_S | 2.63 GB | 3.2 GB | **40 s** | fluent, but semantic slips |
| `gemma-4-12B-it-heretic` | Q4_K_M | 7.4 GB | **8.4 GB** | 60 s | no slips in the sample |

Aggregates do not separate them — both give 17/17 lines, 0 metadata leaks, 0 foreign characters,
identical name handling, Cyrillic share 0.79. The gap is semantic and shows on single sentences.
*"…a tin of licorice drops at his elbow"*:

- 4B: «баночкой с жевательной резинкой на подоконнике» — licorice became chewing gum, elbow
  became windowsill.
- 12B: «баночкой лакричных конфет у локтя» — correct, slightly literal.

*"…Cormac Doyle had owned boats … in the lot behind the boatyard"*: 4B says «владел яхтами …
в гараже за причалом» (yachts, garage); the 12B says «владел лодками … на стоянке за верфью».
*"She set the reel in the tin"*: 4B renders tin as «латунь» (brass); the 12B says «жестяная
коробка».

That is the whole trade: 2.4× the time and 2.6× the VRAM for prose that stops inventing objects.

## Discarded, with the measurement that discarded it

| candidate | why it is out |
| --- | --- |
| `qwen3.8:27b` | **20 GB resident at 32 k — needs both 16 GB cards**, so it cannot run while the next adapter trains. Best Russian of anything measured; kept only as an offline reference. |
| `qwen3.8-27B-abliterated` (`huihui-ai`, Q4_K_M 16.6 GB) | same VRAM problem as its base, plus unvetted third-party weights |
| `gemma4:26b-a4b-it-q4_K_M` (≈15–16 GB) | no headroom left on one card; MoE decode is cheap but the weights still occupy the whole card |
| `qwen3.6:35b-a3b`, `Hermes-4.3-36B` (Q4_K_M **21.8 GB**) | both cards |
| `Mistral-Small-3.2-24B-abliterated` | published as **safetensors only** — no GGUF, so `ollama pull hf.co/…` cannot fetch it |
| `gemma4:latest` (7.5 B), `gemma4:31b` | superseded by the 12B on quality-per-GB; the 31B needs both cards |
| `machinum:0.8b` | the sibling artefact: 1.088 train loss but **eval red** — it does not hold the line protocol, so it has no role |

Two conclusions survived contact with the measurements:

1. **Abliteration is irrelevant here.** It removes refusal behaviour; literary translation never
   triggers one. What the job needs is Russian morphology, register and consistency. The 12B is
   in the list because 11.9 B parameters write better Russian than 4.7 B, not because it is
   uncensored.
2. **VRAM, not speed, is the budget.** The pipeline is line-serial with concurrency 4; a 4B
   already answers faster than it can be scheduled. A "fastest" MoE buys nothing. A model that
   occupies both cards costs the next training run, which is the actual scarce resource.
