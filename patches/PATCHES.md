Patches from a performance audit of a Qwen3.8-27B (6-bit, hybrid GatedDeltaNet + attention) Hermes Agent lane
on a Mac mini M4 Pro 48 GB, macOS 26.6.2, mlx 0.32.2.

Status on 2026-09-10 (mlx-dspark 0.18.1 and mlx-vlm 0.7.0 have since been released):
- dflash_model patch: SUPERSEDED by mlx-dspark 0.18.1 (issue #33 fixed upstream); kept for 0.18.0 users.
- wide_gemm patch: still an alternative on 0.18.1, which instead suspends CPU co-prefill for the session under memory
  pressure (issue #31); the fp32 BLAS route keeps co-prefill working (+15-20 % cold prefill here).
- prefix_cache patch: still applies to 0.18.1 (prefix_cache.py unchanged between v0.18.0 and v0.18.1).
- mlx-vlm apc patch: against 0.6.17; 0.7.0 redesigned APC and keeps make_warm_batch_exact_cache_multi, not re-measured.

mlx-dspark 0.18.0 (unified diffs against the installed package files):
- mlx-dspark-0.18.0-dflash_model.patch — drafter RotatingKVCache single-row append after a partial prefix
  restore (issue #33)
- mlx-dspark-0.18.0-wide_gemm.patch — CPU co-prefill share through Accelerate fp32 BLAS instead of BNNS bf16
  (issue #31)
- mlx-dspark-0.18.0-prefix_cache.patch — rung-ladder inheritance on partial hits, WARN shed keeps rungs,
  checkpoint persistence across restarts (materialized on attach), minimum slot size, deferred file deletion,
  lazy best-slot re-attach after a CRITICAL shed

- mlx-dspark-0.18.1-server_cpu_split_keep.patch — 0.18.1 suspends CPU co-prefill for the session on the first memory-pressure
  shed (its #31 mitigation); with the fp32 BLAS route active that crash cannot occur, so this env-gated 3-liner keeps the
  split on (MLX_DSPARK_CPU_SPLIT_SUSPEND=1 restores upstream behaviour). Validated 2026-09-10 (acceptance/speed identical).

mlx-vlm 0.6.17:
- mlx-vlm-0.6.17-apc.patch — exact APC hit path: keep a single-row warm cache plain instead of merging it into
  Batch* caches (Qwen3_5's singleton shortcut re-merged the whole KV/SSM cache every decode token)
