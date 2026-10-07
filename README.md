# Qwen3.8-27B agent lane for 48-64 GB Apple Silicon

Recipe, measurements and tooling for running an abliterated Qwen3.8-27B as a local agent backend (Hermes Agent in this
case) on a 48 GB Mac mini M4 Pro. Everything here was measured on that one machine (macOS 26.6, mlx 0.32.2) with one
harness. Where a number rests on one seed or a handful of prompts, the text says so.

## Start here

The setup serves the model fully locally as an OpenAI-compatible endpoint on `127.0.0.1`. Hermes Agent is the client
here, but any OpenAI-compatible client works. You need Apple Silicon with at least 48 GB of unified memory and about
26 GB of disk for the weights (~22 GB) and the drafter (~3.6 GB).

Headline numbers. The two output figures are both correct; they measure different things:

| Figure | What it measures |
|---|---|
| 23-28 tok/s | Decode only (server counter), short prompts, 512-token answers |
| 17-23 tok/s | Whole request timed end to end, real agent prompts of 2.6k-26k tokens (table in [Speed](#speed)) |
| 143-160 tok/s | Cold prefill, prompts of 2.6k-20k tokens |
| 11-12 tok/s | Plain decode without the drafter, for comparison (bandwidth-bound) |

Which file you want:

| You want to | Use |
|---|---|
| Run the fast text server | `start-dspark-backend.sh` plus the patches: [Reproduction](#reproduction), steps 1-3 |
| Stop empty answers when thinking uses up the token cap | `thinkbudget_proxy.py` (standalone, non-streaming requests only; streaming passes through unchanged) or `thinkbudget_sidecar.py` (inside a proxy you already run; handles streaming too) |
| Connect Hermes Agent | `hermes-provider.yaml` |
| See what each patch changes and which mlx-dspark version it applies to | [`patches/PATCHES.md`](patches/PATCHES.md): four mlx-dspark patches (three apply to 0.18.1) and one mlx-vlm patch, also as a [gist](https://gist.github.com/felix-ab/78ab16b7d15ba32432e9d8d972cd5ca9) |
| Know why each setting was chosen | [Recipe](#recipe), [Speed](#speed), [What did not help](#what-did-not-help) |

## How we got here

The starting point (late August 2026) used the same weights and 6-bit quant on mlx-vlm 0.6.17 with plain decoding:
11 tok/s at short context, 6.4-6.7 tok/s on cached turns at 30-33k, and 262-272 s to the first token of every new agent
session (a 30.5k-token prompt). The hardware was near its ceilings (decode ~11 tok/s from memory bandwidth, prefill
~130 tok/s from compute), so every gain below is structural. Changes in order, each output-preserving:

1. mlx-vlm cache hit-path fix (`patches/mlx-vlm-0.6.17-apc.patch`): cached turns were copying the whole KV cache on
   every token. Decode at 30-33k: 6.5 to 10 tok/s.
2. Prefix priming plus cache snapshots on disk, done in the client-side proxy: first token of a new session 262 s to
   1.6-2.3 s. That proxy is specific to this machine and not included here.
3. A smaller always-on agent prompt (fewer tool schemas): 30.5k to 15.9k tokens to prefill on any cache miss.
4. Text moved to mlx-dspark with the DFlash2 drafter: decode 11 to 23-28 tok/s at short context and 10 to 19-30 tok/s
   at 30-33k (server counter). mlx-vlm had been slow with drafters because its exact verifier has no 6-bit kernel.
5. Automatic swap to mlx-vlm for requests with images, about 14 s each way (also in the machine-specific proxy).
6. CPU co-prefill through Accelerate fp32 BLAS (`wide_gemm` patch, plus `server_cpu_split_keep` on 0.18.1): cold
   prefill +15-20 %, to 143-160 tok/s.
7. Prefix-cache persistence fixes (`prefix_cache` patch): re-prefill after a new day or a memory edit 130-260 s to
   2-12 s.
8. Budget forcing (`thinkbudget_*`) with thinking at `medium`: empty answers at the token cap 1 in 60 turns to 0 in 90.
9. mlx-dspark 0.18.1 with one RAM cache slot, since two or three slots caused swapping. A later local patch that keeps
   up to three extra checkpoints on disk brought a 24.6k-token conversation back in 2.55 s instead of 167 s; that
   patch is not published here yet.

Related on Hugging Face: our earlier (August 2026) build on a different abliterated body,
[VisualInference/Qwen3.8-27B-AEON-Ultimate-Multimodal-MLX-6bit](https://huggingface.co/VisualInference/Qwen3.8-27B-AEON-Ultimate-Multimodal-MLX-6bit)
and its [MTP drafter](https://huggingface.co/VisualInference/Qwen3.8-27B-AEON-Ultimate-MLX-MTP-Drafter). Their
status sections carry the same-harness comparison to this lane and the faster serving path for that build.

## Recipe

| Piece | Choice | Reason |
|---|---|---|
| Weights | [`orcarouter/Qwen3.8-27B-Uncensored-MLX`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX), `6-bit/` (affine, group 64, ~22 GB, vision tower bf16) | Best published capability table among the abliterated Qwen3.8-27B bodies. On our long-session harness OrcaRouter, Heretic-ARA and stock Qwen are within two-seed noise, so no body showed an advantage. 4-bit disagrees with 6-bit on 11 % of next tokens and is not faster at agent context sizes. 8-bit leaves no room for the drafter and prefix cache in 48 GB. |
| Precision map | uniform 6-bit | A sensitivity-driven mixed 6/8 map (315 modules at 8-bit, +2.5 GB) cut KL to the 8-bit reference by 21-56 % depending on the metric and changed nothing measurable in 40-turn retention, tool use or drift at two seeds. It cost ~6 % decode. Minima (arXiv 2609.04098) reports the same pattern: protect-GDN maps win perplexity, not tasks. |
| Engine | mlx-dspark 0.18.1 + [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (block-diffusion drafter, lossless verify) + three patches | 15-19 tok/s end-to-end at 24-27k context on this body. mlx-vlm with the model's own MTP head: 6-7 tok/s. Plain decode is bandwidth-bound at 11-12 tok/s (22 GB over 231-273 GB/s). |
| Draft width | `--max-draft auto` in production. Pin an integer for any A/B. | The derived width is depth-adjusted from a per-drafter calibration curve. A freshly calibrated drafter got 2 drafts per round at 24k where the stock one got 3, which turned one of our head comparisons into a width comparison until we noticed. |
| Prefix cache | one slot, 4,096-token rungs, checkpoints persisted to disk | With 2-3 slots at 38k context the machine swapped ~1 GB during a 27k generation. One slot costs nothing for a single user. Persisted checkpoints cut a returning 24k prefill from ~240 s to ~70 s across restarts. |
| Thinking | on, `reasoning_effort: medium`, T 1.0, top_p 0.95, top_k 20, presence 0, 4,096-token completion cap | `xhigh` produces the empty-answer-with-stop failure (Qwen3.8 issue #216, 19-38 % of turns). At `medium` we saw none in ~600 turns. Reasoning replay across turns (`reasoning_echo`) measured worse: late retention down, context doubled, speed -18 %. |
| Budget forcing | `thinkbudget_proxy.py` or `thinkbudget_sidecar.py` | At the 4,096 cap, 1 of 60 open-ended turns ended inside `<think>` with no answer across two seeds; at 2,048, 4 of 30. When a turn ends with `finish_reason=length` inside the think block, the proxy re-renders the prompt, appends the partial reasoning and `</think>`, and continues on `/v1/completions` for up to 2,048 answer tokens. The shared prefix hits the cache. With the proxy: 0 of 90 turns. Unaffected requests pass through unchanged (131 checked byte for byte). |

## Speed

Final configuration: `--max-draft auto`, one prefix slot, thinking on at medium, T 1.0. Cold prefill is a fresh server's
first request with that prompt (1 token generated). Warm decode is the same prompt again (prefix-cache hit) generating a
512-token thinking answer, timed end to end. Long prompts use a real 15.3k-token agent system prompt plus prose.

| Prompt tokens | Cold prefill tok/s | Warm decode tok/s (end to end, 512 tokens) | Accepted draft tokens per round |
|---|---|---|---|
| 2.6k | 155 | 22.8 | 2.8 |
| 3.2k | 151 | 21.2 | 2.7 |
| 6.8k | 153 | 19.9 | 2.5 |
| 10.2k | 155 | 17.2 | 2.2 |
| 11.8k | 160 | 22.7 | 3.0 |
| 13.9k | 143 | 17.3 | 2.6 |
| 20.1k | 152 | 17.2 | 2.3 |
| 25.6k | cache-assisted (shared an 18.6k prefix with the previous request) | 18.7 | 2.7 |

Cold prefill is flat at 143-160 tok/s from 2.6k to 20k tokens. Decode is 21-23 tok/s below ~12k context and 17-19
tok/s at 14-26k. Acceptance per round varies with the prompt (2.2-3.0).

Draft-width sweep at 24-28k context (6 open-ended turns, thinking on):

| max_draft | Accepted per round | Server decode tok/s | End-to-end tok/s |
|---|---|---|---|
| 3 | 2.50 | 17.1 | 14.2 |
| 4 | 2.66 | 15.7 | 13.0 |
| 5 | 2.75 | 12.1 | 9.8 |
| auto (settled on 2 at this depth) | 2.39 | 17.1 | 14.0 |

Wider is slower at this depth because verification pays a width x depth KV-read cost. `auto` and 3 are equivalent here,
and `auto` widens at shorter contexts (3.0 accepted per round at 11.8k). Acceptance by traffic type at width 3: open
prose 2.5, code 2.9, tool-call turns 3.2, prose with thinking off 2.5.

## Quality

The harness plants 8 instructions and runs 40 scripted turns with tool calls (retention), runs 30 tool tasks over a real
18-tool schema (JSON validity, schema, recovery from an injected error, "stated intent but no call"), and replays a real
~24k-token session followed by 30 open-ended turns (repetition, cap hits, empty answers). Two seeds.

- OrcaRouter 6-bit on this recipe: retention 0.90-0.95 (turns 31-40: 0.91-0.93), tool JSON and schema 100 %, right tool
  0.78-0.89, no loops (max identical-line repeat 1-2 over 30 turns).
- Stock Qwen3.8-27B, Heretic-ARA and AEON trial-48 at BF16 on an H100, same harness: retention 0.93-0.96, all within
  seed noise of each other and of the 6-bit OrcaRouter runs. AEON had the most answers cut off inside thinking (3-7 %),
  and its card documents long-generation loops.
- KL of OrcaRouter 6-bit to stock 8-bit: 0.0075 nats, top-1 agreement 97.6 %. The abliteration accounts for 0.0053 of
  that and quantization for 0.0032. Measured on 1,820 assistant-role positions of chat-templated real sessions: a small
  sample with a consistent direction.
- The August 2026 public build `VisualInference/Qwen3.8-27B-AEON-Ultimate-Multimodal-MLX-6bit` (AEON
  trial-48, vision 6-bit), same harness, same Mac, same drafter: behaviour indistinguishable at one seed (retention 0.92
  vs 0.90, tool JSON and schema 100 % both, empties 7 % both), but KL to stock 8-bit 0.0875 with p99 1.36 and top-1
  agreement 93.4 %, against 0.0075 / 0.10 / 97.6 % for OrcaRouter 6-bit. That, the 2.5x faster text lane and budget
  forcing are why the recipe moved.

## What did not help

1. Mixed 6/8-bit precision from an activation-weighted sensitivity map over 130k tokens of real agent traffic: KL only.
2. Body swaps: Heretic-ARA and AEON are not measurably better than OrcaRouter on retention, tools or drift.
3. Reasoning replay across turns: worse late retention, twice the context, slower.
4. Native MTP through mlx-vlm: 2.5x slower than mlx-dspark with DFlash2 at 24k and above.
5. Fine-tuning the DFlash2 drafter on the body's own outputs (two rounds, ~25k regenerated conversations, SpecForge
   online capture on 2x H100). An aggressive schedule lost 4-6 % acceptance on prose and code at equal width. A
   warm-start schedule (lr 2e-5 cosine, 32 samples per step) landed at parity everywhere. The stock incoai head is hard
   to beat at this data volume. If you try: pin the draft width in every comparison and check `position_offered` in
   `/metrics`.
6. Learned or calibrated quantization at 6 bits: no published method beats round-to-nearest here, and mlx-lm's docs say
   DWQ does not train at 6-8 bits.

## Reproduction

1. Download [`orcarouter/Qwen3.8-27B-Uncensored-MLX`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX) (the `6-bit/` folder; gated, accept the terms) and
   [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
2. `python -m venv ~/venvs/mlx-dspark && ~/venvs/mlx-dspark/bin/pip install mlx-dspark==0.18.1` (mlx 0.32.2, mlx-lm
   0.31.3). Apply the three patches that apply to 0.18.1 from `patches/`: `prefix_cache` and `wide_gemm` (named 0.18.0; both
   files are unchanged in 0.18.1 and the patches apply cleanly) and `server_cpu_split_keep` (see `patches/PATCHES.md`;
   the `dflash_model` patch is for 0.18.0 only). Upstream reports: [prefix cache](https://github.com/ARahim3/mlx-dspark/issues/36),
   [CPU co-prefill alternative](https://github.com/ARahim3/mlx-dspark/issues/31#issuecomment-5617867067),
   [mlx-vlm exact-APC hit path](https://github.com/Blaizzy/mlx-vlm/issues/2210).
3. `start-dspark-backend.sh dflash auto` (edit the paths at the top or set the env overrides). Health:
   `curl 127.0.0.1:8044/health`.
4. Budget forcing: `python thinkbudget_proxy.py --upstream http://127.0.0.1:8044 --port 8084 --model-dir <6-bit dir>
   --think-budget 4096 --answer-budget 2048`, then point the client at 8084. `thinkbudget_sidecar.py` is the same policy
   as an importable module.
5. Hermes: `hermes-provider.yaml` has the provider and model block. Keep `reasoning_effort: medium`; leave
   `reasoning_echo` off.
6. Image requests: keep an mlx-vlm 0.6.17 lane with `patches/mlx-vlm-0.6.17-apc.patch`. mlx-dspark is text only.

## Attribution and licenses

Weights: OrcaRouter (Apache-2.0) from Qwen/Qwen3.8-27B (Apache-2.0). Drafter: incoai / z-lab DFlash2 (Apache-2.0).
Engine: mlx-dspark (see its license), mlx-vlm and mlx-lm (MIT). This repository: MIT. It contains no model weights,
training data or session content. The harness prompts are not included because they are built from private sessions.

## Limits of these claims

- OrcaRouter is "not worse, and best documented", not proven better than the alternatives.
- 6-bit is "no difference detected at two seeds on this harness", not proven identical to 8-bit.
- Budget forcing removes the cap-hit failure we observed. The continuation has its own finite budget.
- No speed gain from drafter fine-tuning was found.
