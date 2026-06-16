# Streaming multi-rate ACF on ARM Mali (RK3588) — sizing report

This is a clean DSP/GPU sizing problem. The translated spec has a couple of ambiguous points, so let me state how I'm reading it before I give numbers, then draw the graph and size it.

## Working interpretation (assumptions)

- Sample rate Fs = 2 kHz, samples are 64-bit.
- 500 frequency channels, f = 1…500 Hz (1 Hz step). For each f the integration window is T = 1/f, so the group length in samples is N_f = Fs/f = 2000/f (range: 2000 samples at 1 Hz down to 4 samples at 500 Hz).
- The lag is fixed at τ = T/4 → k_f = N_f/4 = 500/f samples. This is a **single-lag** quarter-period autocorrelation per channel (a quadrature / periodicity detector), *not* a full multi-lag ACF — the formula pins τ to one value.
- Output matrix is 500 (freq) × 2000 (time). The "cyclic shift" is the waterfall scroll on the time axis. At 2 kHz, one new 500-element column per sample fills 2000 columns = exactly 1 s of history.
- "32 ACFs simultaneously" = the GPU's concurrent batch width (32 of the 500 channels computed per wave, ~16 waves/column), folded into scheduling. If instead it means 32 independent input streams, multiply the memory/compute figures by 32 — flagged below where it matters.

## Computation graph

```
            ┌──────────────────────────────┐
            │ Input stream                 │
            │ 2 kHz · 64-bit samples       │
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
            │ Input ring buffer            │
            │ 1 s window · 2000 samples    │
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
            │ Per-frequency windowing      │   fan-out ×500 freq
            │ 500 channels · N = 2000/f    │
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
            │ Single-lag ACF kernel        │   16 waves of 32
            │ lag τ = T/4 · GPU batch 32   │
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
            │ Column assembly              │
            │ 500 ACF values per slice     │
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
            │ Circular matrix + output     │   no memcpy — head pointer
            │ 500×2000 · pointer scroll    │
            └──────────────────────────────┘
```

The headline finding is that **this is a compute-trivial, latency- and datatype-dominated problem**, not a TOPS problem. Here is the reasoning.

## Arithmetic load

For frequency f the integration window holds N_f = 2000/f samples and the single quarter-period lag is k_f = 500/f samples. A single-lag correlation over that window costs ≈ N_f multiply-accumulates (MACs). Summing one full 500-channel output column:

> Σ N_f = 2000 · Σ(1/f) for f = 1…500 = 2000 · H₅₀₀ ≈ 2000 × 6.79 ≈ **13,600 MACs per column** (≈ 27,200 ops, counting mul + add separately).

A new column every 0.5 ms (2 kHz) gives **≈ 54 Mops/s per stream** if you brute-force-recompute each window. Multiply by the 32 concurrent ACFs and you reach **≈ 1.7 Gops/s ≈ 0.0017 TOPS**.

If instead you maintain each correlation incrementally as samples arrive (add the entering product, subtract the leaving one — the "maintain ACF as samples arrive" requirement), the per-channel cost drops to ~2 MACs/sample, i.e. **≈ 0.13 Gops/s for all 32**. Either way you are three to five orders of magnitude below the device ceiling.

The one factor that genuinely inflates this is the **64-bit datatype**. Mali-G610 has no native FP64 and emulates INT64 through INT32 pairs, so each 64-bit MAC costs roughly 4–8× a 32-bit one. Applying a pessimistic ×8 still leaves you at **≈ 0.01–0.05 TOPS**. The Mali-G610 MP4 in the RK3588 delivers on the order of **0.45 TFLOPS FP32** across its 4 shader cores, so peak utilization is a few percent at most.

## Resource summary

| Resource | Requirement | RK3588 Mali-G610 MP4 budget |
|---|---|---|
| Compute | ~0.01–0.05 TOPS (with 64-bit penalty) | ~0.45 TFLOPS FP32 → <5% used |
| Cores | 1 shader core suffices for throughput | 4 available (use the others for latency / double-buffering, not throughput) |
| Output matrix | 500×2000×8 B = 8 MB per stream; ×32 streams = 256 MB | shared LPDDR4/5, 4–32 GB → fine |
| Input buffers | ~16 KB ring per stream (1 s @ 2 kHz, 64-bit) + lag history; ~1–2 MB for 32, doubled for ping-pong | negligible |
| Memory bandwidth | column write 500×8 B @ 2 kHz ≈ 8 MB/s; brute-force input reads ≈ 7 GB/s for 32 streams | LPDDR ~20–50 GB/s → comfortable, but watch the read side |

Two implementation rules carry most of the engineering value here:

The **"cyclic shift" must be a pointer/head-index rotation, never a physical row move.** Literally scrolling an 8 MB matrix at 2 kHz would cost ~16 GB/s of pure copy traffic per stream — that alone would saturate the bus for 32 streams. Implement the time axis as a circular buffer; the "rows shift down" semantics become a modular index in the display read.

The **64-bit dynamic range is the real misfit with this GPU.** If the full 64-bit integer range is genuinely needed at the output, plan the datapath deliberately: INT32 products with emulated INT64 accumulation, or FP64 emulation, and validate numerically. If the application can tolerate FP32 (or FP32 accumulation with periodic rescaling), the kernels get dramatically simpler and faster. This decision should be made early because it dictates the kernel structure.

## The actual bottleneck: synchronization and latency

Because the arithmetic is negligible, the binding constraint is the **0.5 ms column cadence versus GPU dispatch overhead.** A Mali OpenCL/Vulkan kernel launch costs tens to ~100+ µs, so one dispatch per column is wasteful and fragile. This is where the "32 ACFs simultaneously" figure earns its place: batch 32 time-slices (or 32 streams) per dispatch, which amortizes launch overhead and yields a ~16 ms processing latency — acceptable for a 1–500 Hz waterfall, but worth flagging if any tight real-time control loop consumes this output. Pair that with double/triple buffering between the input DMA and the GPU queue, and pin the producer thread so samples never stall the buffer hand-off.

## Implementation and deployment estimate

For one experienced GPGPU/DSP engineer working on the RK3588 (OpenCL or Vulkan compute, mainline Mali driver), a realistic schedule:

| Phase | Estimate |
|---|---|
| Design and fixed-point / dynamic-range analysis | 3–5 days |
| Core ACF compute kernels + host harness | 1–2 weeks |
| 64-bit emulated datapath + numerical validation | 3–5 days |
| Streaming infra: ring buffers, circular-matrix scroll, double-buffering, CPU↔GPU sync, real-time scheduling | 1 week |
| On-target integration, timing/latency tuning, deployment | 1–2 weeks |
| Verification | runs in parallel throughout |

That lands at roughly **5–8 weeks** to a deployed, validated build, with the 64-bit datapath and the sync/real-time layer (not the math) being the two items most likely to absorb schedule slip.

## Closing note

One closing engineering note worth raising before committing to the Mali: the RK3588 also carries a separate **6 TOPS NPU**. The task specifies the Mali GPU, which is fine given how light the load is — but if 64-bit integer handling on Mali proves painful, or if these streams share the SoC with heavier work, re-examining the datatype requirement (or the target engine) is the highest-leverage change available.

If you can confirm whether the 32 figure means 32 independent input streams versus a batch depth, and whether the output truly needs full 64-bit integer range, I can tighten the memory and 64-bit-penalty numbers to a single committed figure.
