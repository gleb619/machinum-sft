# Which base model, and why

Re-ranked on 2026-11-08. Constraints that drive it: **one or two 16 GB consumer cards**, a
text-only pipeline, Russian literary prose, and a strict line/XML output protocol. Llama is
excluded by user preference; the 128 K native window is demoted from requirement to bonus
(see `docs/guiding/sft-model.md` §15.2 in the pipeline repo — a 4–8 K window plus
previous-sentence context recovers most of what long context buys).

## Measured inputs

- **GGUF sizes** from the Hugging Face tree API (`unsloth/*-GGUF`).
- **KV cache** `= 2 × layers × kv_heads × head_dim × bytes` per token.
- **Russian tokenizer fertility measured here**, not quoted: each `tokenizer.json` was run
  over the pipeline's Russian chapters (9 463 characters, 1 422 words).

| Model | Params | Native ctx | KV/token | Q4_K_S / Q4_K_M / Q5_K_M | RU chars/token | Licence |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5 9B | 9.65 B | 262 144 | 128 KB | 5.39 / 5.68 / 6.58 GB | 3.08 | Apache-2.0 |
| Gemma 4 12B it | 11.96 B | 262 144 | 384 KB | 6.76 / 7.12 / 8.41 GB | **3.10** | Apache-2.0 |
| Gemma 4 E4B it | 8.0 B (MoE) | 131 072 | 84 KB | 4.84 / 4.98 / 5.48 GB | 3.10 | Apache-2.0 |
| Qwen3.5 4B | 4.66 B | 262 144 | 128 KB | 2.59 / 2.74 / 3.14 GB | 3.08 | Apache-2.0 |
| Qwen3-8B | 8.19 B | 40 960 | 144 KB | 4.80 / 5.03 / 5.85 GB | 2.46 | Apache-2.0 |

Qwen3-8B's tokenizer spends **26 % more tokens** on the same Russian text than Qwen3.5 or
Gemma 4, and its window is 41 K. It stays only as a text-only fallback.

## Training feasibility on 16 GB (Unsloth's published VRAM figures, verified from their docs)

| Model | Method | Documented VRAM | Fits one 16 GB card |
| --- | --- | --- | --- |
| Qwen3.5 9B | bf16 LoRA | **22 GB** | **no** — and Unsloth explicitly discourages QLoRA on Qwen3.5 (quantisation differences) |
| Qwen3.5 4B | bf16 LoRA | **10 GB** | **yes**, with room for 4–8 K sequences |
| Gemma 4 12B | 4-bit QLoRA | not published; 31B QLoRA is 22 GB, so 12B lands well under it | **yes** — verify on the box (`msft doctor`, then a 20-step run) |
| Gemma 4 E4B | 4-bit QLoRA | ~10 GB | yes |
| Qwen3-8B | 4-bit QLoRA | ~8–10 GB | yes |

Two traps in the docs worth knowing before you start:

- **Gemma 4 E2B/E4B share KV state across layers** (`num_kv_shared_layers`). With
  `use_cache=False` — which gradient checkpointing forces — the shared-KV layers recompute K
  and V locally and the loss diverges into garbage. Use the Unsloth-patched path, not a hand
  written loop.
- **All of Qwen3.5 and Gemma 4 are multimodal checkpoints** (`Qwen3_5ForConditionalGeneration`,
  `Gemma4UnifiedForConditionalGeneration`). Text-only fine-tuning works, but data collators
  may still expect image tensors; keep `pillow`/`torchvision` installed and let the model
  processor handle text-only rows.

## Decision

1. **Train on Gemma 4 12B with 4-bit QLoRA** (`configs/gemma-4-12b-qlora.yaml`) — best Russian
   fertility, Apache-2.0, and a 10.9 GB estimate at r32/seq 4096. This is the default config.
2. **Also trainable today: Qwen3.5 4B with bf16 LoRA** (`configs/qwen3.5-4b-lora.yaml`) — the
   same tokenizer as the 9B, 10 GB of VRAM, the cheapest way to iterate protocol and data.
3. **Serve Qwen3.5 9B** for the final artifact if you can rent 24 GB for a single SFT run;
   otherwise the 4B adapter transfers the same tokenizer and template and is the honest
   local-only answer.
4. **Do not train Qwen3.5 9B at 16 GB.** 22 GB bf16 LoRA is a hard wall, and QLoRA on that
   family is against the vendor's guidance.
5. **Do not train Llama.** Excluded by preference; nothing in the measurements argues to
   bring it back now that 128 K is demoted.

Serving notes: text-only means no `--mmproj`; at 128 K Qwen3.5 9B needs `-ctk q8_0 -ctv q4_0`,
Gemma 4 12B is comfortable to ~32–48 K and needs q4_0 KV beyond that.
