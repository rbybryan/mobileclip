# Architecture Round 4 — the first *profiled* pass (2026-07-29)

Four independent expert reviews (VLM architecture / efficient architecture / complexity &
systems / training-time), each briefed with the verified code, the measured numbers, and the
full dead-list from rounds 1–3 so nothing already-killed got re-proposed.

**What makes this round different from rounds 1–3: somebody finally measured.** Rounds 1–3
reasoned from an *analytic* MACs estimate. This round profiled the model with
`torch.utils.flop_counter`, timed the ops, and diffed the exported ONNX graph. Two of the
plan's load-bearing premises did not survive contact with those numbers.

> **Reading order.** §0 first — it contains two corrections that change what is worth doing.
> §3 is the actionable part: five checks that cost **zero GPU-days** and settle most of §2.
>
> 🔺 **Then §8 (round 5, 2026-07-30) — the first pass to attack the 86.5% term.** It adds
> P11–P14, corrects two premises this file states elsewhere (**P1 is not blocked by any torch
> pin**; **"stage 3 is 50% of MACs" is really "`layers[2]=24`"**), and refutes one suspect before
> anyone spends time on it. **§8 is partial**: 3 of its 4 commissioned panels died on an API
> quota limit, so the accuracy/MFU/quantization sweeps still owe a round.
> ✅ **Update 2026-07-31: all three were resumed and returned. ROUND 5 IS COMPLETE** —
> **§9** accuracy (§9.0 outranks everything in §8: the R1 bank exists only inside the sigmoid
> path, so C3's shipped combination may be net-negative), **§10** quantization/latency (P2's
> retraction *over*-corrected — it is 0.78 LSB, not "survives"; and **P1 subsumes P3 entirely**),
> **§11** MFU (**three of §8.3's five rungs are worth 0%**, §8.4's temporal-permute dismissal is
> **retracted**, and the realistic ladder is **3–6×** ⇒ 6.6 days → 1.1–2.2 days).
> **Read §9–§11 before acting on §8.**
>
> 🔴 **Status as of 2026-07-30 — all five §3 gates have now been run** (§3.0 for results).
> Two things this round asserted did **not** survive: §0.1's confound suspicion (tested,
> **rejected** — the legs stand) and **§2 P2's INT8 mechanism** (tested, **falsified** — the
> temporal term survives quantization). P2's *recommendation* survives on ablation evidence;
> its *reason* does not. Read §2 P2's banner before quoting anything from that proposal.

---

## §0. Two corrections that change the plan

### 0.1 ~~🔴 GATE-0's legs ② and ③ are probably confounded.~~ ✅ **RESOLVED 2026-07-29 — not confounded**

> ✅ **The decisive check below was run and the confound is ruled out.** Of the 80 comparable
> spatial tensors, **0 are bit-identical** to MobileCLIP-S2's (0 name/shape mismatches). The
> spatial tower **was** trained during the video stage, so the "same init, same optimization"
> premise leg ③ needs **holds**, and legs ②③ stand as written. The analysis below is kept as
> the audit record of a hypothesis that was tested and rejected — it was the right thing to
> check, and checking it is what makes the legs quotable.
>
> ~~⚠️ **One number in leg ③ still does not reproduce.**~~ ✅ **RETRACTED 2026-07-30 — the 6.1%
> is correct and my `gate5` was the broken one.** The two scripts divide by different
> denominators:
> - `temporal_capacity_audit.py:178` uses **`token_mixer.layer_scale` only** (40 tensors) — the
>   gate on the **spatial mixer** branch, the one-to-one counterpart of `layer_scale_1d` on the
>   temporal mixer branch. Median ≈ **1.3e-2** (the figure §2 P2 independently quotes) ⇒
>   7.973e-04 / 1.3e-2 = **6.1%**. ✅
> - `gate_checks.py gate5` (my 2026-07-29 version) pooled **all 80** `layer_scale` tensors,
>   folding in the **block-level** gate that scales the **ConvFFN residual** — a different
>   module with a different magnitude. That inflated the denominator to 3.296e-02 and produced
>   the spurious **2.42%**.
>
> **Leg ③ stands exactly as written.** `gate5` now prints the matched ratio as the headline and
> the pooled one only as a clearly-labelled non-comparable reference.
>
> ✅ **CONFIRMED on the server, 2026-07-30.** `temporal_capacity_audit.py` on
> `mobileviclip_small.pt` prints **ratio = 0.0613**, temporal median 0.0008, spatial median
> **0.0130** — and, decisively, **n = 9920 on both sides** (40 blocks × channels), i.e. the two
> gates are matched channel-for-channel. `0.0008 / 0.0130 = 0.0613`. The whole 2026-07-24 audit
> reproduced bit-for-bit; see `experiments/gate0_results.md` §7.
>
> ✅ **CROSS-VALIDATED 2026-07-30 — the fixed `gate5` was run and now agrees.** Output:
> `layer_scale_1d (时序 mixer) blocks=40 n=9920 median=7.973e-04`,
> `layer_scale (空间 mixer) blocks=40 n=9920 median=1.300e-02` ⇒ **0.0613**, identical to the
> audit to four decimals via an **independent code path**. Spatial median 1.300e-02 also matches
> the `1.3e-2` §2 P2 quoted from the start — **three sources, one number.** The old pooled value
> is still printed, explicitly flagged `勿用于腿③ … 0.0242 … 混进了块级 ConvFFN 门`, so the
> error is now documented inside the tool's own output rather than only in prose.

GATE-0 concluded "the model does not encode temporal direction" from three lines of evidence
(`experiments/gate0_results.md` §8). ROADMAP, round 2, and several docs — including edits made
on 2026-07-29 — describe these as **three independent** probes. **They are not independent in
the way that claim requires.**

- **Leg ① (behavioural)** — shuffling 8 frames costs ≈0 (t2v +0.1/+0.2); repeating one frame
  costs −13. This is a direct measurement of the finished model. **It stands, unaffected.**
- **Leg ② (kernel shape)** — temporal k=3 kernels indistinguishable from a random null
  (antisym 0.5011 vs 0.4999).
- **Leg ③ (gate ratio)** — temporal gate at **6.1%** of the spatial gate, "from the same 1e-5 init".

The problem is leg ③'s "same init" premise. Verified in code
(`models/mobileviclip_small.py:94-112`): under `freeze_vision`, **everything is frozen except
`mixer_1d` / `norm_1d` / `layer_scale_1d` / `temporal_embedding`**. And this repo's own
progress report describes the upstream recipe as exactly that —
「冻结的 MobileCLIP 图像骨干 + 只训练注入的时间分支」(`docs/clip-2025-progress-report.md:184`).

If upstream trained that way, then the spatial `layer_scale` we compare against was learned
across **MobileCLIP-S2's full image pretraining (DataCompDR-1B scale)**, while
`layer_scale_1d` was learned only during a **short video stage**. The 16× ratio then conflates
*"the temporal branch is useless"* with *"the temporal branch received ~100× less gradient"*.
Leg ② is the same confound seen from another angle: a `trunc_normal(0.02)` kernel that barely
moved **stays at the random null**, so "indistinguishable from random" and "barely trained"
are the same measurement.

> **Decisive check — ~1 minute, zero GPU, server-side** (weights are gitignored):
> compare the spatial `layer_scale` tensors in `mobileviclip_small.pt` against MobileCLIP-S2's
> image checkpoint. **Bit-identical ⇒ the tower was frozen ⇒ legs ②③ are confounded** and only
> leg ① is publishable as written.

**What this does and does not change.** It does **not** rescue the temporal branch for
deployment — leg ① plus §1's measurements independently say it is not earning its cost
(§2 P2). What it changes is the *scientific* claim. "Temporal structural reparameterization is
inert **as a design**" is not supported if the branch never got a fair training run; the
supported claim is narrower: **"in these released weights, that branch contributes ≈0."**
Round 2 §6 and any paper text must be re-worded accordingly, and the controlled experiment
that would settle it is §2 P4.

### 0.2 ~~🔴 Training is decode-bound at ~0.67% MFU.~~ ✅ **MEASURED 2026-07-30 — the MFU is real, "decode-bound" is not**

> **Verdict: the dataloader is not the constraint.** Measured with
> `experiments/probe_decode_bound.py` on the 101vector box (**192 cores, 176 usable**):
>
> | probe | measured | implication |
> |---|---|---|
> | warm decode | ~20 ms/clip | decode itself is fast |
> | cold read vs cold decode, **disjoint** file sets | I/O ≈ **0%** of cold decode, cold/warm 1.2× | the mount is **not** slow |
> | loader-only, 1 rank × 6 workers | **132.7 samples/s** | training consumes only **13.9/rank** ⇒ **9.5× headroom** |
> | `num_workers` sweep 1→24 | 7.5 → 324.6 samples/s, **no saturation** | cores are not the limit either |
>
> The input pipeline supplies roughly an order of magnitude more than training consumes, so
> **"53 h/epoch ⇒ 6.6 days per 3-epoch run" is not a dataloader artifact.** The 0.67% MFU below
> stands; what does not stand is the *cause* stated in this section's title and in the fix
> ladder. The remaining suspects are on the GPU side of the loop — kernel-launch overhead and
> low occupancy from a depthwise-heavy mobile backbone (see §1: the temporal branch alone is
> **440 ONNX nodes and 80 full-tensor transposes** for 0.20% of MACs), plus DeepSpeed/NCCL
> overhead. **The settling measurement is a real training run**, not another micro-benchmark:
> raise `num_workers` and re-time `s/it` against the 9.1821 s/it baseline. Command and the
> `nvidia-smi dmon` reading rules (which invert depending on sm%) are in `COMMANDS.md` §3.11 ②.

Arithmetic, re-derived and confirmed — **this part is unaffected**:

| quantity | value |
|---|---|
| total MACs/sample (T=8) | 59.7–64.0 G (analytic / profiled) |
| forward FLOPs | ~119 G |
| × grad checkpointing (`gradient_checkpointing=True`, `mobileviclip_small.py:317`) | ~477 GFLOP/sample |
| measured throughput | 9.1821 s/it @ global batch 512 = **55.8 samples/s** |
| ⇒ per-GPU | 6.66 TFLOP/s on 4×H100 |
| ⇒ **MFU vs ~990 TFLOP/s bf16** | **0.67%** |

~~Independent corroboration from the data side: `num_workers = 6` × 4 ranks = 24 decode workers;
55.8 ÷ 24 = **2.32 clips/s/worker = 430 ms/clip** — a textbook `decord` number. Two independent
routes agree.~~ **Retracted.** Decode measures **~20 ms**, not 430 ms — off by 20×. The
"agreement" was a coincidence of arithmetic: dividing a throughput that is limited by something
else by a worker count happens to land in decord's plausible range. **A number landing in a
plausible range is not a measurement.**

Depthwise-heavy convnets are inefficient, but **8–20% inefficient, not 0.7%** — so something
beyond the architecture is still unaccounted for, and that gap is now the open question.

**Why this still matters as much as any proposal in §2:** "6.6 days per run" is the constraint
that shaped the entire experimental plan — U6's budget ladder, the arm-cutting order, the
decision to run ranking arms for 1 epoch, and arguably G2's measurement crisis (you cannot
afford seeds). A 0.67% MFU means **~100× of headroom exists somewhere**; it is simply not in the
input pipeline. Until the GPU-side cause is found, treat the budget as real.

> 🔴 **Two method errors produced two wrong answers here on the same day. Both are now rules.**
>
> 1. **Comparing against a single baseline judges nothing.** The first pass compared
>    loader-only (**1 rank × 6 workers**) against training (**4 ranks × 24 workers**), got
>    "4× the workers, +14% throughput", and concluded a shared resource was saturated. The two
>    sides had different worker counts and were never comparable. An actual sweep showed clean
>    linear scaling to 24 with no saturation at all.
> 2. **A throughput benchmark must drain the prefetch queue before timing.** DataLoader
>    prefetches `num_workers × prefetch_factor` batches. Warming up with 1 batch and measuring
>    40 meant the 24-worker setting (queue of 48) was almost entirely consuming pre-built work
>    — reported as **43× superlinear**, with per-worker throughput *rising* as workers were
>    added. Per-worker throughput cannot rise; that impossibility is the tell. The probe now
>    warms `2 × workers × prefetch_factor` and flags superlinearity in its own output.

Fix ladder — **all rungs below are now off the critical path**, kept for the record: ~~`num_workers`
6→12–16~~ → ~~pre-decode to a 4×4 frame sprite JPEG in WebDataset tars~~ → ~~NVDEC/DALI~~.
They would speed up a pipeline that already has 9.5× of headroom.

---

## §1. The first real profile (supersedes the analytic estimate)

`torch.utils.flop_counter`, reparameterized (inference-mode) tower, 256², B=1.

**MACs by op class, T=8 (total 63.97 G):**

| op class | share | scaling |
|---|---|---|
| **ConvFFN 1×1 (pointwise)** | **86.5%** | O(T) |
| attention qkv/proj | 5.25% | O(T) |
| ConvFFN dw 7×7 | 3.31% | O(T) |
| **attention bmm (the only O(T²) term)** | **2.10%** | O(T²) |
| downsample | 1.16% | O(T) |
| stem | 0.81% | O(T) |
| RepMixer dw 3×3 | 0.59% | O(T) |
| **temporal Conv1d** | **0.20%** | O(T) |

Per-frame per-stage: s1 0.77 G, s2 2.00 G, **s3 3.89 G (≈50% of the model)**, s4 1.06 G.

**Three closures follow immediately:**

1. **The "8% attention" figure in `model-architecture-improvements-zh.md` is stage-4 *total*.**
   Only the **bmm** part is quadratic: **2.10–2.25% at T=8**, 4.4% at T=16, 8.4% at T=32.
   ⇒ **Every proposal pitched as "temporal attention is O(T²), let's fix it" is capped at a
   2–4% compute win before it starts.** Linear attention, divided space-time, windowed
   temporal, temporal token merging: closed on compute grounds, permanently, with a number.
2. **86.5% of MACs are per-frame pointwise convs, exactly linear in T.** The only levers that
   move the dominant term are frame count, resolution, and channel width.
3. **The temporal branch is 0.20% of MACs — and 27.8% of the exported graph.** See §2 P2.

**Wall-clock (CPU, 8 threads, T=8, B=1, summed over 40 RepMixer blocks):**

| path | ms |
|---|---|
| temporal branch (permute → conv1d → permute) | **190** |
| spatial dw 3×3 | 68–98 |
| ConvFFN (dw7 + fc1 + fc2) | 790 |

The temporal branch is **0.20% of MACs but 18.1% of mixer+FFN wall-clock — a ~90× divergence**,
and **~75% of its cost is layout (permutes), not arithmetic.**

**Exported ONNX (opset 17, T=8), baseline vs temporal branch deleted:**

| | nodes | Reshape | Transpose | Conv | layout ops |
|---|---|---|---|---|---|
| baseline | **1581** | 203 | 117 | 229 | 340 (21.5%) |
| temporal deleted | **1141** | 43 | 37 | 189 | 100 (8.8%) |

**440 ONNX nodes — 27.8% of the graph — serve a branch that contributes ≈0 accuracy.**
80 of the 117 Transposes are full-feature-map transposes.

> **This is the project's own concrete instance of "MACs ≠ latency."** Going forward, the rule
> for this codebase is: **count full-feature-map transposes, not FLOPs.**

---

## §2. Ranked proposals (deduped across the four reviews)

Ranked by (expected gain × novelty) ÷ (risk + cost). **P1–P5 cost zero GPU-days to decide.**

### P1 · Delete the temporal branch from stages 1–3 — largest *measured* win
Equivalent to `layer_scale_1d = 0` (which makes `w_1d = id`, `b_1d = 0`, exactly). Removes
**440 ONNX nodes (27.8%), 80 full-tensor transposes, 190 ms of 1082 ms CPU mixer path (−18%)**
for −0.20% MACs and (leg ①) ≈0 accuracy. Second-order benefit is larger: **T disappears from
40 of 44 blocks**, so stages 1–3 become literally MobileCLIP-S2's image tower over T·B frames
⇒ "1 checkpoint → N graphs" collapses to *one image graph with dynamic batch* + one small
stage-4/head graph per T. Publishable as the empirical counterpart to GATE-0.
**Decide in 75 seconds:** set `layer_scale_1d = 0` on the released checkpoint, re-run MSR-VTT.
**Kill if** Δr_mean > 1.0 (raw and DSL, same sign). ⚠️ Delete from the *shipped model*, not
from the *research question* — §0.1 means the branch may never have had a fair run (see P4).

### P2 · ~~Unfuse for quantization~~ 🔴 **MECHANISM FALSIFIED 2026-07-30 — the temporal term survives INT8**

> 🔴 **The decisive measurement was run and it refutes this proposal's premise.**
> `gate_checks.py gate5-fused` reparameterizes the released checkpoint and measures the actual
> fused perturbation `|w_1d − δ|` per channel (n=9920):
>
> | p05 | p25 | **p50** | p75 | p95 |
> |---|---|---|---|---|
> | 4.695e-04 | 2.531e-03 | **6.178e-03** | 1.252e-02 | 2.880e-02 |
>
> INT8 round-to-zero threshold is **3.937e-03**. Only **36.2%** of channels fall below it —
> the **median sits 1.57× above** it. **INT8 does not annihilate the temporal branch; ~64% of
> channels survive.** The claim below is withdrawn.
>
> **Why the prediction failed, precisely.** P2 estimated the fused perturbation as `λ·|Δ|`
> using λ (temporal median 8e-4, since confirmed as 7.973e-04) and an *assumed* BN-fused
> depthwise magnitude of **O(0.1–1)**. The measured fused perturbation implies `|Δ| ≈ 7.7`
> — the BN-folded depthwise kernels are **one to two orders of magnitude larger** than assumed.
> The λ figures were right; the term they were multiplied by was wrong.
>
> **What this kills:** (a) the "INT8 annihilates the temporal branch" mechanism; (b) the
> inversion of pillar 2 below — AC5/AC9 are *not* "cashed out at INT8", because the thing they
> add is not below the LSB; (c) the claim in §3 that `layer_scale_1d = 0` is "the exact INT8
> simulation" — it is not, it is strictly more destructive than INT8. (d) The generalizable
> one-liner `SNR ∝ 127·λ` is **wrong as written**: it must be `SNR ∝ 127·λ·|Δ|`, and `|Δ|` is
> the term that actually decides. That correction is the transferable lesson.
>
> **What survives:** the *deployment* conclusion, but now resting entirely on the ablation
> rather than on quantization. Gate ③ says the branch is worth ≤0.82 r_mean (95% CI, both
> protocols) while costing **27.8% of the exported graph** for **0.20% of MACs**. The branch is
> not quantization noise — it is a real computation that survives INT8 and still buys almost
> nothing. That is a *different* and arguably more interesting finding than the original claim.
>
> ⚠️ **Untested half:** `gate5-fused` measured only the temporal kernel `w_1d`. The companion
> claim that the **spatial** mixer runs at "1–3 LSB / ~2 effective bits" is **still unverified**
> and, given how badly the temporal estimate missed, should be assumed wrong until measured.

<details><summary>Original P2 argument (kept as audit record — premise now refuted)</summary>

`RepMixer.reparameterize()` (`:766-777`) fuses to **δ + λ·Δ** — an O(1) identity tap plus an
O(λ) perturbation, in **one** tensor. Per-channel symmetric INT8 sets the scale from the max
element (the identity, ≈1) ⇒ `s ≈ 1/127 = 7.87e-3`, round-to-zero threshold **3.9e-3**. With
measured λ (temporal median 8e-4, spatial 1.3e-2) and BN-fused depthwise weights of O(0.1–1):
~~**PTQ INT8 annihilates the temporal branch exactly**~~ (bit-identical to `layer_scale_1d = 0`)
and leaves ~~**the spatial mixer running at 1–3 LSB**~~, i.e. ~2 effective bits.

> ~~**This inverts pillar 2.**~~ `c2-vs-alternatives.md` §4 proposes **AC5 + AC9** as
> "reparameterization on the temporal axis — free at train time, **cashed in at INT8**".
> ~~Measured, it is cashed **out** at INT8~~ — refuted above. **AC5's acceptance test
> (「验推理逐位不变」) being an fp32 test remains a fair criticism**, but the specific failure
> mode it was said to miss (INT8 silently deleting the added capacity) does not occur.

Fix: deploy **unfused** (`y = x + conv_λ(x)`, so the conv gets its own scale and keeps ~7 bits),
or INT8 only the pointwise convs (86.5% of MACs) and keep the 40 depthwise mixers in fp16 —
the ops that break under INT8 are exactly the ops that are 0.8% of MACs.
Prior art to differentiate: QARepVGG (2022) for RepVGG; Nagel et al. (2019) for depthwise
ranges. **Neither covers RepMixer/FastViT, and nobody covers the temporal-axis case.**

</details>

### P3 · Filmstrip layout — exact, free, zero permutes ✅ *verified numerically here*
The temporal depthwise Conv1d over T is **exactly** a conv2d on a `(B, C, T·H, W)` filmstrip:

```python
strip = x.reshape(T,B,C,H,W).permute(1,2,0,3,4).reshape(B,C,T*H,W)
y = F.conv2d(strip, w1.view(C,1,3,1), b1, padding=(H,0), dilation=(H,1), groups=C)
```

**Verified in float64 in this repo: max abs error 1.3e-15** (roundoff). Frame boundaries stay
aligned through every stride-2 downsample while H is even (64→32→16→8 ✓), and BN sees an
identical element set, so the per-T BN analysis (`plan-review-upgrades.md` §2.2) is unchanged.
**~1.74× on the mixer path (167.6 ms vs 291.9 ms) and −80 transposes, bit-identical on current
weights — no retraining.** Also unlocks widening to full temporal support (kernel 2T−1) for
~1% of MACs. ⚠️ **Must measure on device before committing**: large-dilation depthwise convs
are exactly the op class that can fall off the ANE fast path — which would make this a
FLOP-neutral rewrite that *loses* on latency.
*(A related exact merge — `dwConv1d_T ∘ dwConv2d_HW ≡ dwConv3d(3,3,3)`, rank-1 kernel, err
1.4e-14 — holds too, exact everywhere if the temporal branch is made bias-free.)*

### P4 · The corrected AC5 — the controlled experiment §0.1 demands
Train K parallel (2+1)D branches with **independent, larger gate inits**, **unfreeze the
spatial tower**, and merge all of them into the same 27-tap depthwise 3D kernel at deploy
(P3's algebra makes this exact; a sum of rank-1 kernels is rank-K in the same taps).
**Deploy cost strictly +0.** This removes all three candidate causes of the branch's inertness
at once — rank-1 constraint, shared 1e-5 init, frozen tower — and answers the real question
instead of the confounded one. **Mechanistic prior against it:** the mean-pool readout is
first-order blind to the zero-sum component of any linear temporal mixer, so expect the
diversity-relevant (DC) part to train and the order-relevant (antisymmetric) part not to.
**The only proposal here that costs main-clock time (~6.6 days). ~~or ~1 day after §0.2~~ — §0.2's speedup did not materialise: the dataloader was measured on 2026-07-30 and is *not* the constraint, so budget the full 6.6 days until the GPU-side cause is found.**
Must include the plain single-branch-with-larger-init arm or the result is uninterpretable.
**Pre-register the kill:** gate still <20% of spatial **and** merged kernel energy still below
the INT8 LSB **and** ΔMSR-VTT < 1.0 ⇒ AC5 is a permanent one-line negative.

### P5 · Readout hygiene — the β prior and the annihilated `temporal_embedding`
✅ *Verified in code.* `FastViT.forward` means **un-normalized** frame embeddings
(`:1446-1453`) and `get_sim` L2-normalizes only afterwards (`criterions.py:110-111`). Exactly:

`score(v,q) = β(v) · Σ_t w_t cos(f_t, q̂)`,  `w_t = ‖f_t‖/Σ‖f_s‖`,  **`β(v) = Σ_t‖f_t‖ / ‖Σ_t f_t‖ ∈ [1,T]`**

β is **query-independent**, so it cancels in v2t ranking but acts as an **uncontrolled,
diversity-proportional multiplicative prior on every t2v score** — the headline metric. Nobody
put it there; it is an artifact of pooling before normalizing, and no paper appears to write it
down for frame-mean video-CLIP. Free deliverables: decompose the −13 freeze effect into
information vs β-inflation; and if β-removal moves t2v, ship it (or learn a single scalar on
log β). Separately, **delete `temporal_embedding`** — F2 proves additive codes are annihilated
by a mean readout, and it is the only T-shaped parameter in the tower, so removing it makes
**C2's unseen-T generalization structural rather than an interpolation hack**.
**Both decidable by eval-only runs on the locked checkpoint.**

### P6 · Late interaction at scoring time (MaxSim / LSE) — cheaper to test than it looks
✅ *Verified:* `get_sim` (`criterions.py:94-140`) **already** handles `vision_proj.ndim == 3`
with `agg_method ∈ {mean, max}`. It is dead code only because `encode_vision` collapses T
first. GATE-0 says the value of T is **coverage** (−13 for a repeated frame, ≈0 for shuffling);
a mean — and equally the zero-init `TemporalAttentionPool`, which is a convex combination — is
a **first moment** and cannot express coverage. **Free gate:** evaluate the locked checkpoint
under `agg ∈ {mean, max, LSE(τ), top-2 mean}`. Honest ceiling: CLIP4Clip's `seqTransf` gives
**+1.4 R@1** over mean on MSR-VTT — that is the realistic bar, so this is a section, not a
headline (X-CLIP, ColBERT, TS2-Net own the mechanism). Costs T× gallery storage.

### P7 · Resolution is an unexamined free variable
✅ *Verified:* the vision tower has **no absolute spatial position embeddings** — stage-4 uses
`RepCPE`, whose `spatial_shape=(7,7)` is documented as a **kernel**, not a grid
(`mobileviclip_small.py:898-924`), and the head is `GlobalPool(pool_type="mean")`. **The tower
is exactly resolution-agnostic: changing `img_size` needs no interpolation and no retraining
to evaluate.** 256² was inherited from MobileCLIP-S2's *image* pretraining and never
re-derived for 8-frame retrieval, where the task is coarse-semantic. Since per-frame cost is
~linear in pixels and 86.5% of MACs are per-frame, an iso-accuracy drop to 181² would be a
**2× cut of the dominant term, as a config change**. Sweep the iso-pixel frontier
(256²,T=8) ≡ (181²,T=16) ≡ (128²,T=32) — training-free, and it upgrades Fig-1 from a 1-D
multi-T curve to a 2-D cost frontier. Cite X3D (which searched exactly this axis question for
supervised Kinetics); what is left is the mobile video-**text** case with on-device latency.
⚠️ Most in need of a real device measurement — small tensors can go memory-bound.

### P8 · Image-text replay against the measured ep0→ep1 collapse
The R1 bank's *entire measured mechanism* was blocking the control's ep0→ep1 collapse
(r_mean 90.2 → 85.4, reproduced 3×). That collapse is forgetting of the image-CLIP init — which
supplies **54.82 of the final 61.47**. Mix a T=1 lane in (rung 1: the clip's own middle frame
with the same caption, ~20–30% of steps — free, no new data). Because C2 already made every
temporal op T-agnostic, a T=1 sample costs **1/8** the vision compute. Also gives the R1 bank a
mechanistic competitor: if replay reproduces the bank's benefit, the durable +3.9 is a
**regularization** effect, not a hard-negative effect — worth knowing before carrying the bank
into the 10M final train. Low novelty, high expected value.

### P9 · Statistics pooling: `LN(mean) + W₂·LN(std)`, `W₂` zero-init
The **only** proposal that makes the retrieval loss non-flat in the frame-residual subspace
**without** an auxiliary loss (hence no gaming surface). Mean pooling is a rank-1
permutation-invariant statistic; GATE-0 says the useful structure is permutation-invariant and
**higher-order**, which is exactly the gap. ~0.003% inference cost, zero-init ⇒ starts
identical to baseline. Prior art is real (x-vectors/stats pooling, bilinear/NetVLAD) — **sell
the diagnosis, not the layer.** Top risk, self-identified: `std` is dominated by camera/lighting
jitter and `W₂` learns to zero it. See §4 — this is the proposal the open disagreement decides.

---

## §3. The zero-GPU gate list

> ⚠️ **First server run, 2026-07-29 — the gates silently no-oped, and my first root-cause call
> was wrong.** The heredocs wrote `sd.get("model", sd)`; the released checkpoint's top-level
> wrapper key is **not** `model`, so they fell through to the raw dict and iterated `epoch` /
> `optimizer`. Gate 3 matched 0 keys **and still saved**, producing an `_ablate.pt` that was
> byte-identical to the original — an eval would have "passed" having ablated nothing.
> ~~The checkpoint is reparameterized (inference-mode).~~ **Retracted**: `inspect` on the server
> shows it is **training-mode** — 40 `layer_scale_1d`, 80 `layer_scale`, 4 `temporal_embedding`,
> **0** `reparam_conv*`. All gates now go through **`experiments/gate_checks.py`**, which
> probes the mode first and **raises rather than writing anything when a step matches 0 keys**.
> Commands: `COMMANDS.md` §3.11.
>
> ✅ **λ retraction reversed.** The §2 P2 figures reproduce *exactly* from the released weights:
> `layer_scale_1d` n=9920, **median 7.973e-04, p95 2.762e-03** vs the quoted 8e-4 / 2.8e-3.
> They are verified, not unverified.

### §3.0 Results, first complete run (2026-07-29)

| Gate | Result | Verdict |
|---|---|---|
| ① spatial `layer_scale` vs MobileCLIP-S2 | 80 comparable tensors, **0 bit-identical**, 0 name/shape misses | Spatial tower **was** trained ⇒ GATE-0 legs ②③ keep their same-init premise. **Leg ① and the legs survive.** |
| ③ `layer_scale_1d = 0` (40 tensors zeroed) | r_mean **61.12** raw / **61.95** DSL vs baseline 61.55 / 62.28 ⇒ branch worth **+0.43 raw** (95% CI [+0.03, +0.82], significant) / **+0.33 DSL** (CI [−0.05, +0.72], n.s.) | Keep-threshold was Δ > 1.0. **The entire 95% CI lies below 1.0 in both protocols** ⇒ not a null result but an *equivalence* one: the branch does something, and that something is bounded above by 0.82 r_mean. **Fails.** |
| ④ `temporal_embedding = 0` (4 tensors, norms 0.107–0.150) | r_mean **62.25** raw / **62.95** DSL ⇒ deleting it **gains +0.70 raw** (CI [+0.20, +1.20]) / **+0.67 DSL** (CI [+0.17, +1.15]) — **significant in both** | Keep-threshold was Δ > 0.5. Not merely free to delete — **removing it significantly improves MSR-VTT r_mean under the two-protocol rule.** See the t2v caveat below. |
| ⑤ gate magnitudes | temporal median 7.973e-04, spatial median 3.296e-02, **ratio 0.0242** | See the ⚠ below. |
| ⑤-fused (`\|w_1d − δ\|` after `reparameterize_model()`) | n=9920 channels: p05 4.695e-04, p25 2.531e-03, **p50 6.178e-03**, p75 1.252e-02, p95 2.880e-02. INT8 threshold **3.937e-03** ⇒ only **36.2%** of channels zeroed; median sits **1.57× above** threshold | 🔴 **P2's mechanism is FALSIFIED — the temporal term survives INT8.** See §2 P2 for the rewrite and the root cause (assumed `\|Δ\|` = O(0.1–1); actual ≈ 7.7). |

**Baseline reproduced to the decimal** — raw (`ret_test_emb/`) t2v R@1 **40.8** / r_mean **61.55**,
DSL (`ret_test/`) t2v R@1 **42.5**, both matching the B0 anchors — and `sample_type_test: middle`
makes the eval deterministic, so these deltas carry no run-to-run model noise. The remaining
uncertainty is sampling over the 1000 test videos, which is exactly what the paired bootstrap
measures. The 2026-07-30 per-arm re-run reproduced all three arms **bit-identically**, which is
the determinism claim actually tested rather than assumed.

⚠ **Correction (2026-07-30): the two readouts were mislabelled in the first version of this
table.** `retrieval_utils.py:154-160` returns `(dsl, dsl_T, raw, raw_T)` and `:84-85` maps
position 1 → `ret_test/` and position 3 → `ret_test_emb/`. So **`ret_test/` is DSL** and
**`ret_test_emb/` is raw — and raw is the paper's headline protocol** (`COMMANDS.md` §3.7,
`experiments/README.md` §6.1, `experiments/gate0_results.md` §1 all state this correctly; only
this section and `COMMANDS.md` §3.11 had it backwards). **No verdict changes** — both readouts
move the same direction in both gates. What changes is that the *headline* deltas are the
**larger** ones: gate ③ −0.43 and gate ④ +0.70, not −0.33 / +0.67.

~~⚠ **A number that did NOT reproduce.**~~ ✅ **RETRACTED 2026-07-30 — see §0.1. The 6.1% is
correct; `gate5` was dividing by the wrong denominator** (all 80 `layer_scale`, which pools in
the block-level ConvFFN gate, instead of the 40 matched `token_mixer.layer_scale`). `gate5` is
fixed. The struck-through text below is kept as the audit record of my error.

<details><summary>Original (incorrect) claim</summary>

⚠ ~~**A number that did NOT reproduce.** §0's leg ③ states the temporal gate is at **6.1%** of the
spatial gate. Measured: **2.42%** (medians; p95/p95 gives 1.90%). Neither ratio recovers 6.1%.
The *direction* of leg ③ is unaffected and in fact strengthened — the temporal gate is smaller
than advertised — but **the 6.1% figure needs its provenance found or corrected before it is
quoted in the paper.**~~

</details>

**Lesson worth keeping.** I asserted three times, in three documents, that a repo figure "does
not reproduce" — while my own script was the one comparing incommensurable quantities. The tell
was available the whole time: §2 P2 quotes spatial λ = **1.3e-2**, and `7.973e-04 / 1.3e-2` is
exactly 6.1%. **When a re-measurement contradicts a recorded number, check that both sides are
measuring the same thing before writing "does not reproduce" into three files.**

~~⚠ **The paired-bootstrap inputs were destroyed.**~~ ✅ **RESOLVED, AND THE CI IS IN (2026-07-30).**
The first run sent all three arms to the same `output_dir` and overwrote the ranks; the per-arm
re-run fixed it, and the paired bootstrap (10 000 resamples, seed 0) has now been run on both
protocols. **Measured paired resolution on these arms: R-Sum 95% CI half-width ±2.30–3.00 vs
±14.50–14.70 unpaired ⇒ 4.9–6.3× tighter.** (G4's headline "≈11×, ±1.5 vs ±17" came from a
different, more tightly-coupled arm pair — treat 5–11× as the achievable range, not 11× as a
constant. `experiments/ROADMAP.md` G4 row updated accordingly.)

⚠ **The main metric does *not* clear the two-protocol bar, even though r_mean does.** For gate ④
the paper's headline metric **t2v R@1** (`img_r1`) gains **+1.70 raw, CI [+0.40, +3.10]
(significant)** but only **+1.30 DSL, CI [−0.10, +2.70] (not significant)**. House rule is that
an A/B claim needs both protocols to agree *and* both to clear zero. So the defensible statement
is **"deleting `temporal_embedding` significantly improves r_mean (both protocols)"**, and
*not* "it significantly improves t2v R@1".

**Reading of ③+④+⑤-fused together.** GATE-0 concluded the model encodes no temporal *direction*.
Gate ④ now shows the architecture's one explicit temporal-order parameter is not merely inert but
*significantly counterproductive* on MSR-VTT r_mean, and gate ③ bounds the temporal conv branch
at **≤0.82 r_mean** while it costs **27.8% of the exported graph** (§1) for **0.20% of MACs**.

⑤-fused then removes the explanation we thought we had. The branch is **not** below the INT8 LSB
— 64% of its channels survive quantization. So this is not "a computation too small to represent";
it is **a computation that is faithfully represented, costs a quarter of the graph, and buys
≤0.82 r_mean.** The deployment case for removing it is unchanged and the *scientific* story is
sharper: the temporal machinery is well-formed and nearly useless, rather than numerically dead.

**Two limits, stated before anyone over-reads this.** (a) Zeroing a gate in a *trained* model is
not the same as *training without* the branch — the ablation upper-bounds what the branch does
*in this checkpoint*, which is the right quantity for the deployment/P2 question but not a final
architecture verdict. (b) MSR-VTT captions are largely appearance-driven and are a weak temporal
probe; **`ssv2_mc_val` is the benchmark that can falsify "delete `temporal_embedding`"**, and it
is already wired into `available_corpus`. **Run it before ratifying — this has not happened yet.**

> 🟡 **Downgraded from blocker to nice-to-have, 2026-07-30.** Data is still absent and
> licence-gated (`CHECKPOINTS.md` → "SSv2-MC"), but the *time-arrow* question it was meant to
> settle has now been answered twice from the parameter side, without any new data:
> the static audit (`gate0_results.md` §7–§8, run 2026-07-24 and reproduced bit-for-bit
> 2026-07-30) finds kernel antisymmetry **0.5011 vs a random null of 0.500**, the temporal gate
> at **6.1%** of the matched spatial gate, and `temporal_embedding` **below** the reversal null
> (0.60–0.65 vs 0.706 — *palindromic-leaning*, i.e. structurally unable to encode direction).
> Those agree with the behavioural shuffle (+0.1). The repo's standing decision, taken
> 2026-07-24, is therefore **do not pay to acquire RTime/SSv2 video for the time-arrow
> question** (`notes/order-sensitive-eval.md` §3).
>
> **What SSv2-MC would still add:** it probes action dynamics more broadly than time reversal,
> so it remains the sharpest available check on gate ④ — just not a prerequisite.
> **Gate ④'s verdict is MSR-VTT-scoped**, and now has a mechanistic account for *why* deleting
> an additive, mean-pooling-annihilated, palindromic-leaning positional code costs nothing.
>
> ⚠️ Two traps recorded while wiring this up. (1) SSv2-MC is a **174-way multiple-choice
> accuracy** task — the entry point is **`tasks_clip/retrieval_mc.py`**, not `retrieval.py`;
> the two read the same config and compute different metrics, so the wrong one yields a
> meaningless number rather than an error. (2) `retrieval_mc.py` reduced straight to a scalar
> `acc` and dumped nothing per-question, so an A/B would have had **no confidence interval** —
> exactly the failure the paired bootstrap exists to prevent. It now writes `mc_correct.npz`
> and `collect_results.py --compare-mc` does the paired CI (synthetic check: ±0.72 paired vs
> ±2.99 unpaired, 4.2×). Chance is 0.57%; upstream reference is **acc 46.07**.

All on the **locked checkpoint**, all falsifiable, all cheap. Together they settle most of §2.

| # | Check | Cost | Decides |
|---|---|---|---|
| 1 | **Spatial `layer_scale` in `mobileviclip_small.pt` vs MobileCLIP-S2's** — bit-identical? | ~1 min | **§0.1** — whether GATE-0 legs ②③ are publishable as written |
| 2 | Dataloader-only throughput + SM% during a step | 30 min | ✅ **DONE 2026-07-30** — loader has **9.5× headroom**, so it is **not** the constraint; the 6.6-day budget is real until the GPU-side cause is found (§0.2) |
| 3 | `layer_scale_1d = 0` → re-eval MSR-VTT | 75 s | **P1**. ~~*is* the exact INT8 simulation for **P2**~~ — **wrong, refuted by step 5**: INT8 zeroes only 36.2% of channels, so this ablation is strictly *more* destructive than INT8, not equivalent to it |
| 4 | `temporal_embedding = 0` → re-eval | 75 s | **P5** |
| 5 | Per-channel `max\|Δ\|` histogram vs the 3.9e-3 INT8 threshold | ~10 min | ✅ **RUN 2026-07-30 — refuted P2's mechanism** (36.2% zeroed, median 1.57× above threshold). §2 P2 rewritten |
| 6 | `agg ∈ {mean, max, LSE, top-2}` eval sweep | ~4 GPU-h | **P6** + the §4 disagreement |
| 7 | Iso-pixel grid at 128²/160²/181²/224²/256² (+ per-T BN recal) | ~12 GPU-h | **P7** |
| 8 | Ablation-by-deletion export timing at random init (latency ≠ weights) | 1 day, 0 GPU | the real 92/8 **latency** split |

**Standing rule worth adopting** (the institutional fix for "MACs ≠ latency"): *no training run
is authorized for an efficiency claim until the random-init variant has a measured on-device
number.* Latency does not depend on weight values, so this is always available up front.

---

## §4. The one open disagreement — and the single experiment that settles it

Two reviews read the same T-curve (`r_mean` 55.4/60.0/61.6/62.3 at T=2/4/8/16, fitting
**`r_mean ≈ 63.0 − 15.2/T`**, asymptote ≈63) in **opposite** ways:

- **Variance reduction.** Error ∝ 1/T is the signature of averaging T noisy estimates of one
  video-level embedding. ⇒ the right aggregator **down-weights** outlier frames
  (precision-weighted mean), and the whole aggregator axis is capped at ~1.4 points.
- **Coverage / selection.** GATE-0's −13 for a repeated frame says frames carry *complementary*
  content. ⇒ the right aggregator **up-weights** the best-matching frame (MaxSim, P6).

These prescribe **opposite** mechanisms, and the built `neural_inspired/temporal_surprise_pool.py`
assumes the second (it up-weights surprise). GATE-0's −13 is consistent with **both** and does
not discriminate.

> **The discriminating experiment (one afternoon, cached features).** On MSR-VTT 1k with cached
> per-frame embeddings, compute: **(A)** r_mean from a single random frame (averaged over
> frames); **(B)** oracle best frame per video; **(C)** mean of 8; **(D)** mean of 16.
> **C ≫ A and B ≈ C ⇒ variance reduction.** **B ≫ C ⇒ coverage**, and a learned selector has
> headroom. Bonus: **(B) is a hard upper bound on any content-based frame-selection scheme**,
> which retroactively bounds the entire dead pruning/selection lane — useful in related work.
>
> One-minute corollary: read the sign of `gamma` in the 1M-tier `nA1` checkpoint. **If
> `gamma < 0`, the surprise-pooling module is doing precision weighting and the Rao–Ballard
> predictive-coding narrative in `neural_inspired/README.md` is backwards** — a reviewer will
> find that.

**All four reviews independently proposed the same missing infrastructure**: a **tier-0
frozen-feature cache**. One forward-only pass storing per-frame post-projection embeddings
(`10.65M × 8 × 512` fp16 ≈ **87 GB**, ~3 GPU-days once — minutes per sweep afterwards) turns
every head/aggregator/objective question from "one 6.6-day run per idea" into a same-day paired
sweep at full scale. It is legitimate under the budget rule because it is *one forward pass*.
Known bias: features come from a mean-pool-trained backbone, so it is a ranking signal and a
**lower bound**, biased *against* aggregators that would reshape features (i.e. against P6/P9).

---

## §5. Newly closed — do not re-propose (with numbers this time)

- **"Temporal attention is O(T²), let's fix it."** The quadratic term is **2.10–2.25% of MACs
  at T=8**, 4.4% at T=16. Best case for a heroic attention rewrite is ~4%. Closed on compute
  grounds. (May still be justified on *training memory* — see below — never on compute.)
- **SSM / Mamba / linear attention / gated recurrence on the temporal axis.** At T ∈ [2,16] the
  entire temporal mixing operator is a T×T matrix — at most 256 numbers. Dense is cheaper *and*
  strictly more expressive, with no scan. SSD's advantages are asymptotic in L and start
  mattering in the hundreds. Worse on T2: selective SSMs have data-dependent A/B/C and a scan —
  dynamic control flow, the thing Core ML/TFLite punish hardest. And an SSM's entire structural
  advantage is an **order-sensitive state**, which GATE-0 says is worth ≈0 here.
- **Input-space temporal fusion** (frame stacking, RGB-diff, temporal DCT into "virtual
  frames"). Diversity is a set-**union** property in semantic space; pixels compose by
  **occlusion**, not superposition, so no linear map in pixel space implements union. Averaging
  near-duplicates at 1 fps gives motion blur; differencing gives out-of-distribution input for
  an image-pretrained backbone. *(Exception, flagged: channel-stacked frame groups change the
  **number of passes** rather than transforming the input, so round 2's "no train-time-only
  idea can attack the per-frame cost" does not cover it. Sibling of AC2 — see below.)*
- **Sub-linear in T at constant per-frame fidelity.** Cost is Θ(T) by construction; to beat it
  you must not look at some frames (= smaller T, already on the frontier), look at all frames
  at reduced fidelity (a geometric fidelity pyramid — the one live option), skip by content
  (forbidden by T2), or share computation across frames — which fails because at 1 fps
  *low-level* content genuinely differs (camera motion) while *semantics* barely do, so the
  redundancy sits in the wrong layer to exploit by sharing.
- **Buying more frames.** The fit says T=32 is worth ≈+0.3 r_mean. Above T=8 there is **≤1.4
  points total, forever**. Dead axis.
- **Turning on the SigLIP sigmoid loss for its own sake.** Its advantage is that it *permits*
  large batches; at global batch 512 you are 30–60× below where it pays. Expect ≈0 in isolation.
- **The zero-init attention pool (`r1b`) will not move the needle** — predicted independently
  by two reviews. It is a convex combination = first moment, and cannot express coverage.
  Expect ≤ +0.5 R@1, inside the paired band. Keep as an ablation row; do not budget a full run.

**Correction to AC2's stated risk** (`ROADMAP` 架构探索 backlog): AC2's listed risk is "early
pooling dilutes short actions". **GATE-0 says motion is worth ≈0; the real risk is *diversity*
dilution, worth up to 13 R@1.** Pooling frames to one tensor at depth d is the freeze
experiment applied from depth d onward. That changes both the ablation (ablate on
**scene-diverse** clips, not action clips) and the fix (**retain r=2–4 temporal components,
not 1**).

**Still open, and worth 20 lines:** `use_flash_attn` is threaded from `FastViT.__init__` down to
`AttentionBlock.__init__` and **never read** — `MHSA.forward` is unconditionally naive. This is
irrelevant for export (Core ML/TFLite lower attention themselves) but **not** for training: at
T=16, N=1024, 20 heads, the materialized attention probabilities are what your 23.3 GB peak is
made of, and they gate whether T=16 is trainable at all. ~~torch 1.13.1 has no SDPA and the
sm_70 pin is load-bearing, so the zero-risk fix is **manual query-block chunking inside
`MHSA.forward`**, not a torch upgrade.~~ 🔴 **Retracted 2026-07-30 — the premise is false; see
§8.1. Both GPU boxes already run torch 2.5.1, so `F.scaled_dot_product_attention` is available
today and the manual-chunking workaround is unnecessary work.** Either fix it or delete the
flags — right now they read like a feature and are a trap.

---

## §6. Provenance

Four independent reviews, 2026-07-29, each given the verified architecture, measured results
(multi-T curve, GATE-0, export smoke, 1M tier), the T1/T2 constraints, the 4×H100 budget, and
the rounds 1–3 dead-list. Claims marked ✅ above were **re-verified directly against this
worktree** before landing here — specifically: the filmstrip equivalence (float64, err 1.3e-15),
`get_sim`'s existing `ndim==3` multi-vector path, the mean-before-normalize ordering behind β,
`RepCPE`'s kernel-not-grid semantics, `freeze_vision`'s unfreeze set, `gradient_checkpointing`
defaulting True, and the MACs/MFU arithmetic. **Unverified here** (needs the server, since
weights are gitignored): the λ magnitudes, the CPU wall-clock timings, and the ONNX node counts.
Treat those three as the reviewer's measurements pending replication — checks 1, 3 and 5 in §3
replicate them.

---

## §7. External SOTA cross-check (2026-07-30) — five outside candidates, zero survive

A user-facing assistant session, working from a stale (2026-07-15/24) read of this plan and one
round of its own literature scan, proposed five model-side levers drawn from recent image-CLIP /
VideoCLIP / efficient-VLM literature: a CoCa-style captioning auxiliary decoder, a learned
per-frame adaptive-compute gate (AdaViT/DynamicViT/Mixture-of-Depths-style), register tokens
(Darcet et al., ICLR'24), a FILIP-style token-wise contrastive loss, and feature/attention-map
distillation from InternVideo2. Checked against this round's own kill-list, round-3's C1–C9, and
the T1/T2 constraints before landing anything: **none survive as new work.**

| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| 1 | CoCa-style captioning auxiliary decoder | **Already killed** | `model-architecture-improvements-round3-zh.md` §3: "CoCa 式 caption 解码头:增自回归解码成本、与冻结文本塔冲突,端侧不值" — independently listed in `ROADMAP.md:402-403` and `c2-vs-alternatives.md:101`'s dead-end lists |
| 2 | Learned per-frame adaptive-compute gate | **Killed by standing constraint T2** | `neural-inspired-directions.md` §0 T2: Core ML/TFLite reward static, dense, structured graphs; **data-dependent control flow** is named as one of the three disqualifying failure modes. §5 of *this file* already applies it verbatim: "skip by content (forbidden by T2)". A content-gated per-frame skip is exactly that forbidden case — reviving it means abandoning the mobile-export goal, not a training trick |
| 3 | Register tokens (Darcet et al., ICLR'24) | **Deprioritized, not duplicate** | Distinct mechanism from the removed C4/AC4 "temporal register / latent-bottleneck attention" (`ROADMAP.md:394-395`, dropped for lack of motivation) — but §1 above measures stage-4 attention (qkv/proj + bmm) at **~7.35% of MACs total** (5.25% + 2.10%), and the bmm term any register-token change would sit inside is capped at 2.10–2.25%. Engineering effort there is low-value while P1 (delete temporal branch, an ~18% wall-clock win, already decided) and P7 (resolution cut, 2× the *dominant* 86.5% term) sit unclaimed |
| 4 | FILIP-style token-wise contrastive loss | **Subsumed — defer** | §2 P6 above ("late interaction at scoring time") already covers the motivating mechanism: `get_sim` already supports `agg_method ∈ {mean, max}` at zero build cost, and its own §3 gate (check 6, ~4 GPU-h) is the correctly-sequenced next step. A full FILIP *training-time* loss (token-token contrastive, not just an inference aggregation choice) is materially bigger and unjustified until check 6 shows late-interaction scoring beats mean pooling by more than a section's worth |
| 5 | Feature-level / attention-map distillation from InternVideo2 (MGD-style) | **Mostly already exists** | `UTA_Loss`/`new_UTA_Loss` (`models/criterions.py:409,522`, per `data-quality-caprl-scheme.md:35`) already implement middle+final-layer feature-cosine distillation against a teacher, unwired — the same dead-code pattern P4 wired up for the sigmoid loss. Round-3's C1/C1a/C1b ("attn→conv distillation", kept, rewritten) appears to cover the attention-map-specific extension already; this cross-check could not establish novelty beyond what C1a/C1b already plan and did not re-derive their content |

**One candidate from the same pass survives as a real, non-duplicate gap — but it's a data lever,
not a model lever**, so it's addended to `notes/data-quality-caprl-scheme.md` rather than here:
DFN/MetaCLIP2-style corpus-level filtering (as opposed to per-clip recaptioning) is surveyed in
`docs/clip-2025-progress-report.md` §3–4 but never wired into an execution plan or given a
decision owner.

**Byproduct of the same pass:** this project's own regime — text tower frozen except the final
projection, vision tower open — was mislabeled "LiT-style" in six places (`model-architecture-
improvements.md:71` and its `-zh` twin, `paper-plan.md:292` and its `-zh` twin,
`neural-inspired-round2.md:495-496` and its `-zh` twin). Classic LiT (Zhai et al., CVPR'22) locks
the *image* tower and tunes text — the inverse of this project's setup. Corrected in place;
citations of the *real* LiT paper elsewhere (`docs/clip-2025-progress-report.md`,
`caption-kd-review-2026-07.md`, `recaption-master-plan.md`) were accurate and left untouched.

**Net effect: zero new P/AC-numbered proposals.** The value of this pass is independent
confirmation that this plan's own kill/defer calls on all five externally-sourced candidates
were already correct given the evidence gathered in rounds 3–4 — worth having for the paper's
robustness story, not worth adding to the backlog.

---

## §8. Round 5 (2026-07-30) — the first pass to attack the **86.5%** term

> ⚠️ **This round is PARTIAL and must be read as such.** Four expert panels were commissioned
> (VLM/CLIP accuracy levers · efficient architecture · MFU diagnosis · on-device quantization).
> **Three died mid-run on an API quota limit; only the efficient-architecture panel returned.**
> So §8.3's proposals are one panel's output plus this session's own verification — *not* the
> four-way convergence that rounds 3–4 had. **The accuracy-side, MFU-side, and quantization-side
> sweeps still owe a round.** Do not read "nothing else came up" into this section.
>
> What makes the round worth landing anyway: **§8.1 and §8.2 are corrections and derivations
> verified here**, independent of any panel, and each changes what is worth doing.

### §8.1 🔴 P1 is not blocked. The `torch 1.13.1` premise describes no machine in use.

Multiple docs defer P1 (SDPA/efficient attention) because "the env is torch 1.13.1, which
predates SDPA". **Measured against the repo's own setup docs, that environment exists nowhere:**

| machine | role | torch | source |
|---|---|---|---|
| **b100 / 8×H100 (sm_90)** | main training clock — R0, all ranking arms, winner's final train | **2.5.1+cu124** | `experiments/SETUP_B100.md` §2, measured 2026-07-24: the repo pin 「**从没被装上**」 |
| **101vector / 8×V100 (sm_70)** | eval + export smoke | **2.5.1+cu118** | `ROADMAP.md:126` (pinned *at 2.5.1* deliberately, for the sm_70 kernels), `export_smoke_results.md` §env |

`mobileviclip.yml`'s 1.13.1+cu117 is validated **only on a CPU node** (an import check).
⇒ **`F.scaled_dot_product_attention` has been callable on both GPU boxes since at least
2026-07-24.** Three consequences:

1. **§5's recommended workaround is unnecessary work.** "torch 1.13.1 has no SDPA … so the
   zero-risk fix is manual query-block chunking, not a torch upgrade" — struck in place.
2. **`paper-plan.md` §6.8 item 8's demotion rationale is half-expired.** It priced in
   "DeepSpeed-compat validation risks more schedule than it saves". That validation is
   **already paid**: the 1M-tier arms trained ~15 h under DeepSpeed on 2.5.1 (`ROADMAP` G9),
   and both R1 arms ran on `2.5.1+cu118` (`ROADMAP.md:304`). The surviving leg is narrower than
   written — naive attention is fine at **T=8** (N=512), but at **T=16** (N=1024) the
   materialized probabilities *are* the 23.3 GB peak, i.e. the thing constraining **C2's own
   elastic-T upper bound**.
3. **`torch.compile` / `channels_last` / CUDA graphs are all available** — see §8.3 P12.

**The swap is mathematically exact, verified here in float64** at this model's real stage-4
shape (N = T·H·W = 512, 20 heads): `max|naive − SDPA| = 2.08e-16` (roundoff only). Recipe and a
10-second both-boxes self-check: `COMMANDS.md` §3.13.

⚠️ **Ship it behind `hasattr(F, "scaled_dot_product_attention")`**, so the CPU-node 1.13.1
import baseline keeps working; reuse the existing `use_flash_attn` flag, which is threaded
through the constructors and **never read**.

### §8.2 ✅ Every RepMixerBlock costs the *same* MACs. "Stage 3 is 50%" is a statement about depth.

*Derived from `models/mobileviclip_small.py:295-297` (`layers=[4,12,24,4]`,
`embed_dims=[80,160,320,640]`, `mlp_ratios=[3,3,3,3]`, stem stride 4 ⇒ 64²/32²/16²/8²), and
**re-verified numerically in this worktree**.*

ConvFFN's fc1+fc2 = `2·(3C²·S)`. C doubles while S quarters, so **3C²S is invariant across
stages**:

| stage | C | S | **fc1+fc2 per block** | depth | stage total | MAC share | **byte share** (fp16 act. I/O) | **arith. intensity** |
|---|---|---|---|---|---|---|---|---|
| 1 | 80 | 4096 | **157.29 M** | 4 | 0.629 G | 8.2% | **24.2%** | 60 FLOP/B |
| 2 | 160 | 1024 | **157.29 M** | 12 | 1.887 G | 24.5% | **36.3%** | 120 |
| 3 | 320 | 256 | **157.29 M** | 24 | 3.775 G | 49.0% | **36.3%** | 240 |
| 4 | 640 | 64 | **157.29 M** | 4 | 0.629 G | 8.2% | 3.0% | 480 |

40 RepMixerBlocks + 4 AttentionBlocks × 157.29 M = **6.921 G/frame = 89.6%** of the 7.72 G/frame
per-frame total — i.e. this reconstructs §1's *measured* 86.5% pointwise share from first
principles, and the per-stage totals match §1's measured 0.77/2.00/3.89/1.06 G to within the
non-ConvFFN remainder. **One block = 2.04% of per-frame MACs, in every stage.**

**Three consequences that reorder the plan:**

**(a) "Stage 3 = 50% of the model" is not about stage 3. It is about `layers[2]=24`.** MACs here
are a pure function of *depth allocation*; there is no per-stage width story in the MACs column.
This makes **depth a first-class lever that no prior round considered** (§8.3 P11).

**(b) Resolution beats a narrower student, and the roofline says so before any experiment.**
H100 fp16 ridge ≈ 989 TFLOP/s ÷ 3.35 TB/s ≈ **295 FLOP/byte**; Apple ANE ≈ **230**. Stages 1–3
sit at 60–240 ⇒ **memory-bound on both the training GPU and the target device.** Intensity
`C·N/(C+N)` is independent of S and of depth, so **resolution and depth are
intensity-preserving** (MACs→latency ≈1:1), while **width is intensity-destroying** (C→C/√2 cuts
FLOPs 2× but bytes only √2×). ⇒ **round4-P7's primacy is confirmed on a stronger argument than
"86.5% and linear in pixels", and "distill into a narrower S1/S0" is dominated.**

**(c) The MACs profile misranks cut targets on-device.** Stage 1 is 8.2% of MACs but **24.2% of
bytes** — one stage-1 block moves **8× the bytes of a stage-4 block for identical MACs.** This is
the same lesson as §1's "count transposes, not FLOPs", now quantified for the dominant term.

> ⚠️ **Estimate, not measurement:** the byte-share and intensity columns are analytic
> (activation I/O, fp16, ignoring fusion). A fused ConvFFN would move less. They are strong
> enough to *rank* levers and not strong enough to quote as latency.

### §8.3 New numbered proposals

Numbering continues §2's P-series (P1–P9 are taken). **All costs assume the standing
4×H100 / 6.6-day-per-run budget.**

#### P11 · Latency-weighted structured **depth** pruning — decided by a training-free gate sweep
*NetAdapt (ECCV'18, arXiv:1804.03230) for latency-table-driven pruning; Gromov et al.
arXiv:2403.17887 and ShortGPT arXiv:2403.03853 for layer-drop sensitivity probes.*

**Mechanism — and it needs no code change.** Verified in code: `RepMixer.forward:743-746` is
`x = x + λ_1d·(mixer_1d − norm_1d)` then `x = x + λ·(mixer − norm)`, and
`RepMixerBlock.forward:1075` is `x = token_mixer(x); x = x + drop_path(layer_scale·convffn(x))`.

> 🔴 **Correction to the panel's version, which said "two tensors".** Making block *i* an exact
> identity on the **released checkpoint** requires **three** gates:
> `blocks[i].token_mixer.layer_scale_1d` (temporal mixer) · `blocks[i].token_mixer.layer_scale`
> (spatial mixer) · `blocks[i].layer_scale` (block-level ConvFFN gate). Omit `layer_scale_1d`
> and the temporal branch still runs — the block is **not** an identity.
>
> ⚠️ **This is exactly the 40-vs-80 distinction that already produced one wrong number in this
> repo** (§0.1/§3.0: `gate5` pooled all **80** `layer_scale` = 40 spatial-mixer + 40 block-level
> ConvFFN gates, inflating a denominator and inventing a spurious 2.42%). The block-drop harness
> **must** zero the matched pair per block and assert the count, or it will repeat that error.

So the sweep is **40 RepMixerBlocks** (stages 1–3), reusing the *existing* `gate_checks.py`
harness at ~75 s/eval ⇒ **~1 GPU-hour for the full per-block sensitivity curve, zero GPU-days.**
Stage 4's 4 `AttentionBlock`s have a different gate structure (`:1111-1188`) and are
**unverified** — check before including them. Cross the curve with per-block on-device latency
(available at **random init**, per §3's standing rule) and remove greedily by Δr_mean/Δms.

- **Measured:** each removed block = **157.29 MMACs = 2.04% of per-frame MACs**, and
  5.24/2.62/1.31 MB per frame of activation traffic depending on stage.
- **Estimated (soft):** 4–8 stage-3 blocks removable for <1.0 r_mean after a short heal ⇒
  8–16% MACs. **The estimate is soft; the sensitivity curve is not — it costs one GPU-hour and
  either exists or doesn't.**
- **Mandatory control:** dropping a block shifts every downstream BN's input distribution. **Run
  forward-only BN recalibration before each eval** (`experiments/per_t_bn.py` already exists and
  is the same machinery round4-P7's sweep needs) or you are measuring distribution shift, not
  block utility.
- **Not on the kill-list:** every killed item attacks the temporal axis (0.20%) or attention
  (7.35%). This is the first attack on per-frame **depth**, i.e. on 100% of the 86.5% term. It is
  **not** content-based skipping — blocks are deleted statically at export, identically for all
  inputs ⇒ **T2-clean** (dense, static, shape-invariant).
- **Kill:** if no single block costs <+1.0 r_mean, or the cheapest 6 sum to >2.0 r_mean, depth is
  not redundant in this checkpoint — close the lane and put the budget into round4-P7.
- **Paper or engineering:** **both, and it is the strongest paper item this round.** "Where does
  a distilled mobile video-CLIP put its compute, and what does each block buy?" plus §8.2's
  MACs-uniform/bytes-non-uniform result — a defensible cost-frontier figure **with zero training
  runs**.
- **Sequencing note:** run P11 **× round4-P7 as a 2-D grid, not a union.** Both are
  intensity-preserving (§8.2b), so their latency savings compose and the iso-latency frontier is
  real.

#### P12 · The MFU ladder — with its top suspect already refuted here
**Standing:** §0.2 leaves "the GPU-side cause of MFU 0.67%" as the open question, and it is worth
more than any accuracy item because it sets the 6.6-day clock.

✅ **Settled here, in one command — the panel's #1 suspect is wrong.** It proposed that 9.1821
s/it might be an **8×V100** number (the config carries `batch_size = 64  # 8x V100-32GB`,
`config.py:21`), which would make MFU 2.7% rather than 0.67%. **Refuted by arithmetic:** 1M × 3
epochs ÷ global 512 = 5859 it × 9.1821 s = **14.94 h** vs the reported `Training time 15:11:58`
= **15.20 h** (ratio 1.017, the remainder being eval/checkpointing). **9.1821 s/it is the 4×H100
run. MFU 0.67% stands.**

**The bandwidth side of the roofline, never computed before (analytic estimate):** per-sample
ConvFFN activation traffic ≈ 86.6 MB/frame × 8 ≈ 693 MB; ×~2.5 for BN/GELU/residual/permutes
≈ 1.7 GB; ×4 for fwd+recompute+bwd ≈ 6.9 GB/sample. At 13.95 samples/s/GPU ⇒ **~96 GB/s ≈ 2.9%
of H100 HBM3.**

> **0.67% of compute peak and ~3% of bandwidth peak ⇒ the model is neither compute- nor
> bandwidth-bound. The GPU is idle or running pathological kernels.** That inequality halves the
> hypothesis space for free.

**Settling measurement (~30 min):** `torch.profiler` with `ProfilerActivity.CUDA` over ~20
steady-state steps; compute **Σ(kernel durations) ÷ wall-time = GPU busy fraction.** Busy ≈ 1 ⇒
bad kernels (rungs 1–2). Busy ≪ 1 ⇒ host-side stall (rungs 3–4). Note §0.2's dataloader probe
does **not** rule out a host stall: it measured worker output rate in a *separate process*, not
the main loop's consume path (shm deserialization of a ~400 MB batch + H2D + DeepSpeed syncs).

> 🔴 **This ladder was audited 2026-07-31 and THREE OF ITS FIVE RUNGS ARE WORTH 0%.**
> `cudnn.benchmark` is **already `True`**; the fp16 loss scaler was **measured at zero
> overflows**; `find_unused_parameters` is **dead code** because DeepSpeed never constructs DDP.
> The `channels_last` estimate is inflated ~2×. **Read §11 before acting on any row below** —
> §11.2 gives the corrected ladder, and §11.1 shows the real suspect is the one §8.4 dismissed.

Ladder, all grounded in this worktree, cheapest first:

| # | Suspect | Evidence here | Est. win | Effort |
|---|---|---|---|---|
| 1 | **No `channels_last` anywhere** — grep of `models/ tasks_clip/ utils/ dataset/ tasks/` returns **zero hits**. A pure convnet in fp16 **NCHW** on H100 forces non-tensor-core kernels or implicit layout transforms around all 229 convs | verified | **2–4×** (NVIDIA convnet guidance) | 2 lines |
| 2 | **`cudnn.benchmark = len(train_media_types) == 1`** (`tasks_clip/pretrain.py:263`) — **False** whenever the run mixes media types, leaving cuDNN on heuristic algo choice for 229 convs. This model has 4 fixed spatial shapes | verified | 10–30% | 1 line + confirm which branch the live run takes |
| 3 | **fp16 + DeepSpeed dynamic loss scaling** (`use_bf16=False`) — every overflow **discards a completed fwd+bwd**, invisible in s/it. Read `skipped_steps` in the DeepSpeed log; bf16 is free on H100 and removes the failure mode (fp16 is a *V100* constraint being paid on the H100 box) | verified in config | 5–50% if thrashing | 1 flag |
| 4 | **`find_unused_parameters=True`** (`:279`) with `freeze_text=True` (63M frozen) — forces full autograd traversal per step, defeats bucket fusion. **First check whether the DeepSpeed path bypasses DDP entirely** — if so it is dead code and should be deleted so it stops reading like a feature | verified | 10–40% | 1 line |
| 5 | **Selective checkpointing.** `gradient_checkpointing=True` **is** needed (unckeckpointed peak ≈ 40–80 GB/GPU at batch 64 × T=8) but costs only 4/3, not 100×. Checkpoint **stages 1+2 only** — 60% of activation bytes, 33% of MACs (§8.2) ⇒ overhead 33% → ~12% | estimate | **~+18%** | ~1 day |

⚠️ **Ordering constraint:** `channels_last` must land **after** round4-P1 (delete temporal
branch). The temporal `reshape → permute(1,3,4,2,0) → reshape` chain (`:726-731`) assumes NCHW
contiguity; `.reshape` on a channels_last tensor silently inserts a copy.

⚠️ **MFU bookkeeping:** the 0.67% numerator counts recompute as useful work, which *flatters* it.
Against 3×119 GFLOP the honest figure is **0.50%**.

**Kill:** if GPU busy-fraction >0.9 **and** rungs 1–3 together yield <1.5×, the architecture
really is this inefficient on H100, the 6.6-day budget is physics, and the lane closes with a
number.

#### P13 · The text tower's **25.3M-parameter embedding table** — 0% of FLOPs, training-free to shrink
*ALBERT factorized embeddings (ICLR'20, arXiv:1909.11942) for the retrain variant; BPE
vocabulary/merge pruning for the training-free one.*

Verified from `models/backbones/internvideo2/mobileclip/configs/mobileclip_s2.json`:
`vocab_size=49408`, `dim=512` ⇒ token embedding = **25.3M params contributing exactly zero
MACs**, vs ~37.8M for the 12-layer body. With `max_txt_l=32` the text tower is ~63% of model
params and ~2% of MACs. Two training-free cuts: **(a)** quantize/palettize the table alone (pure
lookup, no accumulation ⇒ error is a bounded per-token perturbation; Core ML supports this
natively): 50.6 MB fp16 → 25.3 MB INT8 → ~12.6 MB INT4 with per-row scales. **(b)** Prune the BPE
vocab + merge list to tokens the train/eval corpora actually produce — deleting a merge is
*exact* on covered text. **Estimated** 12–20k of 49,408 tokens cover >99.9% of video captions at
≤32 tokens ⇒ 25.3M → ~8M. Combined: **50.6 MB → ~4 MB ≈ −27% of total model params, zero GPU.**

- **Not a duplicate of P8** (killed as a paper item: "pure compression, no novelty"). P8 replaces
  the transformer **body** and needs distillation; this is the **table** and needs nothing. They
  compose. **P13 is engineering — ship it, footnote it, do not build a section on it.**
- **Kill:** kill (b) if corpus coverage at the chosen vocab size is <99.9%, or if MSR-VTT r_mean
  moves at all (75-s eval, existing harness). Kill (a) at whatever bit-width first moves r_mean
  beyond the paired band.

> 🔺 **Strategic question this raises, worth more than the megabytes — and it currently has no
> owner.** In a retrieval deployment the **vision** tower runs once per gallery video (amortized,
> offline, arguably server-side), while the **text** tower runs on every query in the interactive
> path. If that is the product, the entire 86.5% analysis is about **indexing throughput**, not
> **query latency**, and P8's priority should go **up**. MobileCLIP/MobileViCLIP report
> *image-encoder* latency, so the paper's headline is the indexing one. **Someone should write
> down in one sentence which of the two C4's latency table measures** — it silently decides
> whether P11 + round4-P7 or P8 + P13 is the main efficiency result.

### §8.4 Considered and rejected on the numbers (round 5)

| Candidate | Why it dies |
|---|---|
| **Inference-time low-rank factorization of fc1/fc2** (C→r→3C) | FLOPs ratio 4r/3C, but **bytes ratio (4C+2r)/4C > 1 always** — the rank-r intermediate is a new full-spatial write+read. At r=0.5C: FLOPs 0.67×, **bytes 1.25×**. Stages 1–3 are memory-bound (§8.2b) ⇒ **it makes the model slower.** Legitimate only as a *train-time-only* branch (P14 below) |
| **Grouped / blockwise pointwise convs + channel shuffle** (ShuffleNet-style) | FLOPs ÷g, bytes unchanged ⇒ **intensity ÷g**, the worst trade in a memory-bound regime; shuffle adds traffic (8C·S → 14C·S = **1.75×**) and **is a full-feature-map transpose** — self-rejecting under this repo's own "count transposes" rule |
| **Structured channel-**width** pruning of the 1×1s** | Width is intensity-destroying (§8.2b): 2× the FLOPs cut buys only √2× latency. Strictly dominated by P11 (depth) and round4-P7 (resolution) |
| **Unstructured / N:M sparsity on the pointwise convs** | T2. 2:4 is a training-side H100 feature with no on-device counterpart here |
| **Distilling into MobileCLIP-S1/S0** | (i) It is a width cut ⇒ dominated (§8.2b). (ii) S1/S0 have **no video-stage checkpoint**, so you pay image *and* video stages, ≥2 runs ≈ 13 GPU-days, to land on a frontier MobileCLIP already published. round4-P7 is a **config change**. ⇒ **The roofline decides this against the smaller student before any experiment** |
| **Shrinking dw 7×7 → 5×5/3×3** | 3.31% of MACs, and depthwise sits at intensity ≈ k² (49→25 FLOP/B), far below every ridge ⇒ removing FLOPs you were never paying for. **≈0 latency win** |
| **MobileOne/RepVGG over-parameterization of RepMixer's dw3×3** | 0.59% of MACs — same category error as AC5: right technique, wrong term |
| **Cross-block weight sharing (ALBERT-style) in stage 3** | Saves params, saves **zero** FLOPs and zero bytes. Vision tower is 36M — params are the *text*-side problem (P13), not the vision-side one |
| **`torch.compile` / CUDA-graph capture as the MFU fix** | Now **available** (§8.1 corrects the panel, which thought it was pin-blocked) — but **still not the story**: ~4,400 kernels/iteration at 9.18 s/it ⇒ launch overhead ≈ **0.4%**. Keep it as a later multiplier, not a suspect |
| ~~**Chasing the temporal branch's 80 permutes for MFU**~~ 🔴 **RETRACTED 2026-07-31 — this row is wrong, see §11.1** | ~~Costed: ~43 GB permute traffic ×3 ≈ 129 GB ⇒ ~38 ms of 9,182 ms = 0.4%.~~ **The costing was wrong three ways**: it divided by *peak* bandwidth (these are the least-coalesced kernels in the model), it counted only the permutes and not the 4 BN1d + 2 Conv1d + ~7 elementwise ops that also live in the transposed domain, and it used the **ONNX** transpose count when in eager PyTorch there is **no transpose node at all** — the cost hides inside `clone` and inside the shape of every downstream kernel. **Measured: the branch is ~33% of training memory traffic and 15 of 33 full-activation kernels per block.** It is the ladder's biggest rung, not a non-issue |
| **Winograd / FFT convolution** | 1×1 convs — 86.5% of the model — gain nothing from either |

#### P14 · (parked, low confidence) Train-time-only over-parameterization *inside* ConvFFN
*ExpandNets (NeurIPS'20, arXiv:1811.10495); Diverse Branch Block (CVPR'21, arXiv:2103.13425);
RepOptimizer (ICLR'23, arXiv:2205.15242) to mitigate train cost.*

`ConvFFN.forward` (`:874-881`) is `dw7×7 → BN → fc1 → GELU → fc2`: **dw7×7, its BN, and fc1 are
three adjacent linear ops with no nonlinearity between them**, so any linear over-parameterization
inserted there folds away at export (e.g. `M` between BN and fc1 collapses as `fc1' = fc1·M`;
a low-rank branch collapses as `fc1' = fc1 + UV`). **Exported graph byte-identical in shape;
inference cost exactly +0**, and it is free in *both* roofline regimes.

**Parked rather than proposed, on the panel's own self-critique:** DBB's published +0.7–1.9
IN-1k comes substantially from *receptive-field diversity across branches*, and `fc1`/`fc2` are
1×1 — a sum of full-rank matrices spans the same function class as one, so any gain there is a
pure optimization/implicit-regularization effect (Arora et al., ICML'18), plausibly small. **The
genuine-diversity half is the dw7×7 = 3.31% of MACs.** Train-time cost is the binding problem at
6.6 d/run (DBB ≈1.5–2× conv wall-clock; RepOptimizer removes it only for constant-scale,
BN-free branches). ⇒ **Only worth running if P11 finds blocks worth removing** — its entire
justification is buying back removed capacity. Pre-register that the fc1/fc2 half may come back
null; that is a cheap publishable negative.

*One thing to say explicitly in any writeup: **P2's falsified INT8 pathology does not transfer
here.** That failure mode is specific to RepMixer's fused `δ + λ·Δ`, where an O(1) identity tap
sets the quantization scale and starves the perturbation. `fc1`/`fc2` have no identity tap; the
collapsed matrix is a single O(1)-scaled weight and quantizes normally. A reviewer who read P2
will ask.*

### §8.5 Sequencing and provenance

**Order (respects the dependencies; steps 1–3 and 5 total ≈1 GPU-hour + ~1 week engineering):**

1. **P12's profiler run** (~30 min) — GPU busy-fraction splits the hypothesis space. *(P12's
   suspect #1 is already refuted above; do not re-spend time on it.)*
2. **P11's 40-block drop sweep** (~1 GPU-hour, zero GPU-days) — produces the analysis figure and
   the pruning decision at once. **Include BN recalibration and assert the 3-gates-per-block count.**
3. **round4-P1 (delete temporal branch — already decided) → then P12 rungs 1–4.**
   `channels_last` must follow P1.
4. **round4-P7 × P11 as a 2-D (resolution × depth) iso-latency grid.**
5. **P13** whenever someone has two spare days. **§8.1 (P1/SDPA)** is ~20 lines and unblocks the
   T=16 memory ceiling; do it alongside.
6. **P14** last, and only conditional on P11.

> 🔺 **Much of this does not need the H100 — and two of §8's own "unverified" items can be
> closed on the idle V100 box today.** See [`v100-10k-advance-tests.md`](v100-10k-advance-tests.md),
> which tiers every item by what it *requires*. Specifically: **P11's identity-collapse premise
> is provable on random input with no eval data** (tier A4 — if a gate is missing it costs 40
> GPU evals to find out the expensive way), and **the spatial-mixer half of `gate5-fused`**,
> which §2 P2 flags as unverified and which decides whether **AC9 gets a new rationale or gets
> buried**, is pure checkpoint statistics (tier A5). P13's BPE-coverage *estimate* also becomes a
> *measurement* there (tier B1). ⚠️ The whole of **P12 is H100-only** — 0.67% MFU is a 4×H100
> number and the V100 has no bf16.

**Provenance.** One expert panel (efficient architecture), 2026-07-30; three sibling panels
(VLM accuracy / MFU / quantization) **failed on an API quota limit and still owe a round**.
Everything marked ✅ or "verified" was re-derived **directly against this worktree** in the same
session: the two-machine torch reality and the SDPA float64 equivalence (§8.1); the per-block MAC
invariance and its reconstruction of the measured 86.5% (§8.2); the 9.1821 s/it provenance
arithmetic, the absence of `channels_last`, and `pretrain.py:263`/`:279` (P12); `layers`/
`embed_dims`/`mlp_ratios`, the three-gate identity collapse and the 40-vs-80 `layer_scale`
hazard (P11); `vocab_size=49408`/`dim=512` (P13). **Explicitly still analytic estimates, not
measurements:** all byte-share / arithmetic-intensity figures, the bandwidth-side roofline, the
BPE coverage fraction, and every "estimated" win in the P12 ladder.

---

## §9. Round 5, accuracy/objective panel (2026-07-31) — the round §8 said was still owed

*One of the three panels that died on a quota limit; resumed and returned. The MFU and
quantization panels are still outstanding.*

### §9.0 🔴 The finding that outranks every proposal: **the R1 bank exists only inside the sigmoid path**

**Verified line-by-line in this worktree, and it changes a shipped GO decision.**

| fact | evidence |
|---|---|
| `vtc_loss_sigmoid` **has** `text_bank=` | `criterions.py:235`; consumed `:279-281` via `_text_bank_nll` (`:286`), which adds **`F.logsigmoid`** terms using sigmoid's `t`/`b` |
| **`vtc_loss` — the softmax InfoNCE path the locked baseline actually trains with — has no `text_bank` parameter at all** | `criterions.py:172`; signature is `(vision_proj, text_proj, idx, temp, all_gather, agg_method)` |
| The only branch that can reach the bank is the sigmoid one | `mobileviclip_small.py:238` `if self.use_sigmoid_loss:` … `else:` → `vtc_loss(...)` with no bank. The docstring at `:245` says it outright: *"R1 **SigLIP-regime** negative pool"* |
| **Every R1 arm at every tier is sigmoid-vs-sigmoid** | 10k long: `nR1_textbank_pilot_long.py:15` vs control `pilot_10k_r1_long.py:13`, both `use_sigmoid_loss=True`. 1M: `sub1m_nR1_s42.py:31` vs control `sub1m_r1both_s42.py:26`, both True — and `sub1m_nR1_s42.py:3` says its **only** addition is `criterion.text_bank`. `ROADMAP` itself labels the control 「SigLIP 制度无 bank」 |

⇒ **The measured +3.9 R-Sum is a delta *within* the SigLIP regime**, and SigLIP is separately
measured as **significantly negative vs softmax** (1M `r1a` vs `r0`: DSL r_mean −0.95
[−1.68, −0.20], R-Sum −5.70, both protocols). So the shipped C3 combination (sigmoid + bank)
is plausibly **net-negative against the locked baseline**, and **the one configuration that
could rescue it — softmax + bank — has never been run, because the code cannot express it.**

> ⚠️ **Do not add +3.9 and −5.70.** They come from **different tiers** (10k × 40 epochs vs
> 1M × 3 epochs), and the three-tier rule says the 10k tier **cannot rank arms**. "≈ −1.8" is
> *indicative arithmetic, not a measurement.* **The way to settle it is to run the softmax+bank
> arm**, which is ~40 lines: `_init_text_bank`/`_sample_text_bank` (`mobileviclip_small.py:168-219`)
> already implement the pre-projection cache, per-step re-projection with the live projection,
> `idx` positive-masking and `dedup_cos` — they are simply wired to the wrong loss.

This is the cheapest high-stakes item in rounds 4–5 combined: **~1 hour of reading to confirm,
zero GPU.** `ROADMAP`'s R1 status row has been amended accordingly.

### §9.1 Reframe: the model is **transfer-limited**, not data-saturated — and one free test decides it

Three measured numbers, read together: image-init floor → final = **54.82 → 61.47** (the video
stage buys +6.65); **1M → 10M data buys only +0.82**; and the control **degrades ep0→ep1**
(r_mean 90.2 → 85.4, reproduced 3×).

A fitting-limited model improves with data (it doesn't). A capacity-limited model shows
train/val divergence (the signature here is worse — degradation *on the eval distribution*).
One mechanism explains all three: **InternVid-10M-FLT captions are machine-generated, short and
appearance-tagged; MSR-VTT/DiDeMo captions are human-written ⇒ the eval set is OOD w.r.t. the
training objective.** More InternVid ⇒ better InternVid fit ⇒ ≈0 transfer; longer training ⇒
more drift off the image manifold; anything constraining drift ⇒ a win *regardless of its
stated purpose* (which is exactly the R1 bank's measured mechanism).

> **Free falsification, from files that may already exist:** if any run logged a held-out
> **InternVid** val loss, plot it against the ep0/ep1 MSR-VTT numbers. **Held-out InternVid loss
> still falling while MSR-VTT degrades ⇒ transfer-limited, confirmed at zero cost.**

**Unstated consequence for C1:** if the binding constraint is *distribution match*, recaptioning
is a **distribution-shift** lever, not a "quality" lever — and the field's reflex (denser, longer,
ShareGPT4V-style) is the **wrong direction** for retrieval against short human queries. Target
~10–20 words in human descriptive register, and follow **LaCLIP** (NeurIPS'23, arXiv:2305.20088):
its winning recipe is **randomly sampling among caption variants per step**, not replacing and
not concatenating (its own ablation shows concatenation underperforms). Keep a substantial
fraction of original captions in the mix. Free sharpening of a plan item that has a script and
zero embeddings extracted.

### §9.2 Survivors (ranked)

**A · WiSE-FT / LiNeS weight-space interpolation to the image init — eval-only, 0 GPU-days.**
*Wortsman et al., CVPR'22, arXiv:2109.01903; LiNeS, ICLR'25, arXiv:2410.17146.*
θ(α) = α·θ_video + (1−α)·θ_S2 over the shared spatial tensors — gate ① already established all
80 comparable tensors exist and differ, so the map is complete; α=0 is essentially the 54.82
floor. Sweep α ∈ {0,…,1} on the **existing 75-s `gate_checks.py` harness ⇒ ~15 min**. The LiNeS
variant makes α depth-dependent and yields a **per-depth forgetting profile** that tells later
arms which stages to open. Published: WiSE-FT gains multiple OOD points at ≈no in-distribution
loss; LiNeS retains 99.8%/97.9% target/control. **Estimate here: +1 to +4 R-Sum, with a real
chance of 0.** Inference cost **exactly zero** (one static checkpoint, α=1 is bitwise baseline).
**Not on any kill-list — weight-space merging appears in no round, and grep confirms no
`model_ema`/soup/interpolation code exists.** **Kill:** if argmax α = 1.0, or the interior α
fails either protocol / its CI includes 0. **That single free gate also de-motivates C, P8 and
the LoRA variants at once.** Engineering win, not a paper — unless α\* turns out ordered across
benchmarks by caption style.

**B · Port the bank to the InfoNCE path and scale K′ to 64k–1M.**
*LiT arXiv:2111.07991 (frozen tower ⇒ precompute); MoCo arXiv:1911.05722 as the contrast.*
Because the text tower is frozen except a 512×512 projection, **every** caption embedding can be
cached pre-projection and re-projected with the live projection each step — **exact, never stale,
no momentum encoder.** 64k × 512 fp16 = 67 MB; the extra logits ≈ 17 GFLOP/step against
477 GFLOP × 512 samples ⇒ **numerically free**, on a run at 0.67% MFU. **Estimate +1 to +4
R-Sum with genuine downside ~0** (the measured evidence says this model is *not*
contrastive-signal-limited) — but it is nearly free and it is **the experiment that resolves
§9.0**. **Not killed:** kill-item #1 killed the *sigmoid link function*, not *extra negatives*;
the code shows the two were never separable. **Kill:** 1M, 1 epoch, K′ ∈ {0,16k,128k}; kill on
the house rule, or if ΔR-Sum is **non-monotone in K′** (that signature means false negatives
dominate ⇒ raise `dedup_cos`, not K′). **The most novel item here** — "a frozen text tower turns
the batch-size axis into a table lookup, giving exact 128k-way contrastive at batch 512 on 4
GPUs" inverts the premise SigLIP exists to solve, and is publishable **even as a negative**.

**C · Anchor distillation to MobileCLIP-S2's *own* image space.**
*LwF arXiv:1606.09282; ZSCL ICCV'23 arXiv:2303.06628; L2-SP ICML'18 arXiv:1802.01483.*
Two rungs: **L2-SP** (`λ‖θ_spatial − θ_S2‖²`, one AXPY/step, λ=0 bitwise-identical,
`checkpoints/mobileclip_s2.pt` already on disk) then a **feature anchor**
(`λ(1 − cos(f_student, f_S2))`, extractable inside C1's existing offline pass ⇒ marginal cost ≈0).
**This is not C1:** InternVideo2 is a *different embedding space* and a **capability** teacher —
it cannot protect the image init and may pull away from it. Same run, different ablation row.
**Estimate +1 to +3 R-Sum.** **Kill:** λ ∈ {0.1,1.0} at 1M; kill on the house rule **or if it
fails to beat Survivor A's free lerp** — if a 15-minute state-dict interpolation buys the same
R-Sum, the 2.2-day arm is not warranted.

**D · SemDeDup core-set selection, riding on C1's teacher pass.**
*SemDeDup arXiv:2303.09540; D4 arXiv:2308.12284; JEST arXiv:2406.17711.*
C1's pass already produces a video embedding for all ~10.65M clips — SemDeDup's exact input.
k-means, drop within-cluster near-duplicates, **plus a video-specific step LAION has no analogue
for: dedup by source-video id** (InternVid clips are cut from the same YouTube videos, which is
why `get_mask` exists at all). **Estimate +0.5 to +2 R-Sum at equal steps, or the same accuracy
in ~half the steps.** The stronger deliverable is free: **"random 1M ≈ full 10M" plus a dedup
histogram converts the project's most awkward null into a data-centric result** about
InternVid-10M-FLT's effective size. Orthogonal to the DFN gap §7 recorded with no owner (DFN
removes *low-quality* pairs; SemDeDup removes *redundant* ones, and saturation says redundancy
binds).

### §9.3 🔴 The §4 disagreement is settleable in an afternoon — **and it may retro-invalidate how the −13 is described**

Literature **cannot** discriminate variance-reduction from coverage: both give `a − b/T` to
leading order. It is an identifiability problem, not a gap. **But P5's own β already answers it.**
For unit-norm frame embeddings with mean within-clip pairwise cosine ρ:

  **β = Σ‖f_t‖ / ‖Σ f_t‖ ≈ √( T / (1 + (T−1)ρ) )**

- ρ ≈ 0.9 at T=8 ⇒ **β ≈ 1.05**, ~1.2 effective directions ⇒ **variance reduction**; the
  aggregator axis is capped at ~1.4 points exactly as the fit says, and **P6/P9/FILIP should all
  be dropped.**
- ρ ≈ 0.5 ⇒ **β ≈ 1.33**, ~2.7 effective directions ⇒ **coverage is real**; MaxSim/top-2 has
  headroom and P9 gets a prior.

> 🔴 **And it retro-explains GATE-0's −13, possibly fatally.** If ρ is high, one frame's
> *direction* is within cos ≈ 0.97 of the mean — **a −13 R@1 cannot be a direction/information
> effect at that geometry.** It would have to come from the β term, which collapses from
> β(v) ∈ [1,T] to **exactly 1** when all eight frames are identical. I.e. **the −13 may be
> substantially a scoring artifact of pooling-before-normalizing, not evidence that frames carry
> complementary content** — and it has been cited as the latter across five rounds, including in
> the brief that produced this section.

**Cost to settle: one afternoon, zero GPU, ~8 MB** — MSR-VTT-1k cached features only (1000×8×512),
**not** the 87 GB tier-0 cache. Compute the distribution of within-clip `cos(f_t, f_s)`,
`cos(f_t, mean)`, per-video β, and each clip's participation ratio; then **re-run the freeze
ablation with the score divided by β** to split the −13 into information loss vs β-inflation.
**Do this before the 4-GPU-h P6 sweep, which it gates**, and before spending on P9 or FILIP.

### §9.4 Considered and rejected (accuracy round)

| Candidate | Why it dies |
|---|---|
| **SogCLR / iSogCLR** (ICML'22, arXiv:2202.12387) — the textbook small-batch contrastive fix | Its per-anchor moving average `u_i` needs **many visits per sample**; you run **3 epochs** (1 for ranking arms) ⇒ `u_i` never leaves initialization and it degenerates to in-batch InfoNCE plus bookkeeping. **Strictly dominated by Survivor B**, which is exact and stale-free |
| **MoCo-style momentum queue on the video side** | Needs a second 36M vision tower + momentum update ⇒ ~+35% step time on the *only* expensive tower, at 6.6 d/run, for **stale** negatives |
| **Temperature sweep / learnable τ** (`temp=0.01`, a plain scalar) | Not worthless — softening τ *is* anti-forgetting — but one scalar costs a 2.2-GPU-day arm, novelty 0, and it is **confounded with every lever above**. Run as a 2-point sanity arm *after* Survivor A, never as a first ranking slot |
| **Unfreezing text / LoRA on text** | The text tower is the only part currently **immune** to the measured failure mode, and InternVid captions are the OOD distribution ⇒ predicted **negative in sign**. One deliberately-negative ablation row at most |
| **Semantic hard-negative mining from teacher embeddings** | The R1 mechanism is measured to be anti-forgetting, not hard-negativeness; InternVid caption redundancy means this **maximizes the false-negative rate** exactly where captions repeat |
| **DaWin** (arXiv:2410.03782) / per-sample dynamic weight interpolation | Computes interpolation coefficients **per test sample** = data-dependent control flow over the *weights*. **Disqualified by T2.** Static α is fine; dynamic α is not |
| **Full online JEST** | Needs a super-batch scored by learner *and* reference every step; published gains are **from-scratch at 40B examples**, not a 3-epoch fine-tune. Do Survivor D's offline rung first |
| **LoRA / surgical fine-tuning as standalone lanes** (arXiv:2405.09673; arXiv:2210.11466) | Good argument (target delta is only +6.65 ⇒ "learn less, forget less" is cheap; LoRA merges at export for +0) but they are the **structural member of Survivor C's family**. **Folded in**: use LiNeS' depth profile to choose which stages to open |
| **Caption ensembling at eval** | In retrieval the caption **is** the query — there are no templates to ensemble. Non-applicable |
| **MobileCLIP2-S2 init swap** (architecture-identical, drop-in) | Possibly the largest single available delta, but costs a **full 6.6-day R0 re-run** and breaks comparability with the replication target ⇒ belongs in the final-model run, not the ranking arms. Note it would give Survivor A **two** image-space endpoints to interpolate toward |

### §9.5 Sequencing (everything before step 5 is zero GPU-days)

1. **Reconcile §9.0 with `r1-implementation-plan.md` §9.4's own write-up of the +3.9** — the code side is settled here; ~1 h to make the prose state the SigLIP-regime scope. **Highest value in rounds 4–5.**
2. **Plot any logged held-out InternVid val loss vs the ep0/ep1 MSR-VTT numbers** (§9.1) — confirms or destroys the reframe from existing files.
3. **Survivor A's α sweep + LiNeS depth profile** (~15 min of eval) — gates C, P8 and the LoRA variants; may ship a free win.
4. **The β/ρ decomposition on cached MSR-VTT-1k features** (§9.3, one afternoon, 8 MB) — gates P6/P9/FILIP **and audits how the −13 is described in the paper**.
5. **Survivor B** (port the bank to `vtc_loss`; K′ ∈ {0,16k,128k}) → **C** (L2-SP first) → **D** (selected-1M vs random-1M).

**Provenance (§9).** Accuracy/objective panel, resumed 2026-07-31. **Re-verified directly against
this worktree before landing:** the `text_bank`/`vtc_loss` asymmetry and the dispatch branch
(§9.0), all four R1 configs' `use_sigmoid_loss=True`, the absence of any EMA/soup/interpolation
code, and `temp` being a plain scalar. **Not verified here:** every quantitative gain estimate,
the β/ρ algebra's applicability to this checkpoint (that *is* §9.3's measurement), and the
published table values cited.

---

## §10. Round 5, on-device quantization / latency panel (2026-07-31)

*The second of the three resumed panels. **MFU is the only one still outstanding.***

### §10.0 🔴 P2's retraction **over-corrected**. The honest statement is "under one bit", not "survives".

**Verified arithmetic here.** `gate5-fused` compared `|w_1d − δ|` against a *fixed* 3.937e-3 and
reported "only 36.2% zeroed ⇒ the branch survives". Re-read the **same quantiles in LSB units**
(LSB = 1/127 = 7.874e-3; round-to-zero at 0.5 LSB = 3.937e-3 ✓):

| p05 | p25 | **p50** | p75 | p95 |
|---|---|---|---|---|
| 0.06 LSB | 0.32 LSB | **0.78 LSB** | 1.59 LSB | 3.66 LSB |

The median channel codes the **entire** temporal perturbation in **0.78 LSB**, and ~72% of the
9920 channels code it in **0 or 1 LSB** (36.2% → 0, ~36% → 1). Effective resolution
**b_eff ≈ 0.7 bits of 8.**

⇒ **All three statements the project has made about this are now orderable:** "INT8 annihilates
it" (original) — **false**. "The temporal term survives INT8" (the retraction) — **true but
badly under-stated**; it survives at *sub-one-bit* resolution. The correct claim is
**"INT8 codes the temporal branch at under one bit."** That is a *better* finding than either,
and **AC9 does not need burying yet — it needs re-founding on this.**

**Corrected law** (replaces §2 P2's `SNR ∝ 127·λ·|Δ|`, which was right in form but had no
reference point):

  **n_LSB = 127·λ·|Δ|_max ⁄ max|δ + λΔ| ≈ 127·λ·|Δ|_max**,  **b_eff ≈ b − log₂(1/(λ·|Δ|))**

### §10.1 ✅ **P1 deletes every temporal Conv1d in the model — so P3 has nothing left to rewrite**

**Verified here by arithmetic that closes exactly.** `layers=[4,12,24,4]` ⇒ stages 1–3 hold
**4+12+24 = 40** RepMixerBlocks (= the 40 `layer_scale_1d`), and their channel sum is
**4·80 + 12·160 + 24·320 = 9920** — *exactly* the n gate5-fused measured. Stage 4 is
`AttentionBlock`s and has no temporal Conv1d at all.

⇒ **round4-P1 removes all 40 temporal branches.** Therefore **P3 (the filmstrip rewrite) is a
contingency for "P1 is rejected on accuracy", not a live proposal** — and the ANE
large-dilation question that P3's ⚠ flags is **off the critical path**. Do not spend on-device
time on it until P1's fate is known.

*(Answered anyway, for the contingency: **the panel's call is that large-dilation depthwise does
NOT stay on the ANE fast path, ~85% confidence.** Mechanistic argument, which is stronger than
the empirical report it cites: your dilation is `(H,1)` with H = 64/32/16/8, so an NPU line
buffer would need `d_h·(k_h−1)+1` = **129 live rows of a 512-row filmstrip at stage 1** — a
different dataflow, not a parameter tweak. Accelerators that support dilation support small
ones (2–4) via `space_to_batch`. **Failure mode is worse than a mild regression**: the conv
falls to CPU/GPU, forcing an ANE↔CPU round-trip **per block, 40× per forward**. If ever tested,
test the exact shapes and read back **per-op device assignment** via `MLComputePlan` (iOS 17+) —
total latency won't tell you why. Stage 3 (dilation 16) has a materially better chance than
stage 1 (dilation 64).)*

### §10.2 ✅ Root cause of the 1–2-order-of-magnitude `|Δ|` miss — verified, and transferable

`Δ = mixer_1d.reparam_conv.weight − norm_1d.reparam_conv.weight` (`:766-777`). **Verified in
code:** `norm` and `norm_1d` are built with **`use_scale_branch=False, num_conv_branches=0`**
(`:659-686`) — i.e. they have **no conv branches at all**; reparameterized, they are a
**pure BatchNorm folded into a kernel**.

⇒ **Δ is dominated by γ/σ, a BN normalizer — not by a conv weight.** O(0.1–1) is the right prior
for a conv weight; **O(1–10) is the right prior for γ/σ** when activation variance is small.
The team applied a conv-weight prior to a BN-scale quantity, and got |Δ| ≈ 7.7 instead of ≈0.5.
**Write this down — it generalizes to every reparameterized block in the MobileOne/FastViT
lineage.**

> ⚠️ **Free footgun worth one grep:** if `reparameterize_model()` ever runs on an **fp16** module,
> `δ + λ(mixer − norm)` subtracts two O(7.7) tensors (fp16 ULP ≈ 7.8e-3) and scales by λ ≈ 8e-4.
> **Assert fp32 at reparameterization time**, cast afterwards.

### §10.3 The recipe — and why the "unfuse for quantization" fix is rejected

| Option | Verdict |
|---|---|
| **PTQ, per-channel symmetric weights, fp16 activations** | ✅ **Do first.** The 86.5% pointwise has no identity tap and quantizes normally |
| **Mixed precision: INT8 pointwise, fp16 the 40 dw3×3 + 40 dw7×7** | ✅ **Second, only if arm 1 loses.** Leaves 3.90% of MACs in fp16 |
| **Deploy unfused** (§2 P2's proposed fix) | ❌ **Reject — quantified below** |
| **QARepVGG branch-scale constraints** | ❌ **Reject on mechanism** |
| **LSQ/PACT/RepQ QAT** | ⏸ **Last**, and not authorized until PTQ is *measured* to lose end-to-end |

**Why unfusing is rejected** (§2 P2 proposed it as the fix; it costs more than the disease):
graph **+160 nodes = +14%** of the post-P1 1141-node graph, in the currency §1 says to count;
activation traffic **+20–25% with epilogue fusion, +70% without**, in stages that are
memory-bound (60–240 FLOP/byte vs ANE ridge ~230) so that traffic is latency ~1:1. **And it does
not even work:** unfused, `mixer(x)` and `norm(x)` are two comparable-magnitude activations whose
*difference* is the signal — you move catastrophic cancellation from the weight domain into the
**activation** domain, where INT8 error does *not* cancel. All to protect a branch bounded at
**≤0.82 r_mean that P1 deletes** (§10.1).

**Why QARepVGG's mechanism does not transfer** (this is the paper-differentiating point):
QARepVGG's pathology is that RepVGG's identity branch carries **its own BatchNorm**, giving a
heavy-tailed fused weight distribution *and* depth-growing activation variance — an
**activation-range** problem. RepMixer is structurally the opposite: the identity is
`mixer.id_tensor`, an **exact δ with no BN**, and λ actively *suppresses* the branch.
**RepMixer's failure mode is branch starvation, not branch explosion.** Applying QARepVGG's
fixes would treat the wrong disease.

> 🔴 **AC9 novelty warning.** Two papers already occupy the space AC9 was going to claim:
> **RepQ** (BMVC'23 oral, arXiv:2311.05317 — QAT treating test-time weights as a differentiable
> function of train params) and **PTQ for Reparameterization via Coarse & Fine Weight Splitting**
> (arXiv:2312.10588 — splits the fused weight *precisely because one component dominates the
> range*, i.e. your identity-tap problem, already published). **AC9's remaining defensible
> novelty is the temporal axis and the measurement, not the technique.**
> *(Also flagged: QARepVGG's arXiv id 2212.01593 was not verified this session — check before citing.)*

### §10.4 The next measurements — and the cheap one is not the histogram

🔺 **Run this first: a 4-arm fake-quant sweep through the existing 75-s eval + paired bootstrap**
(~1 h code, ~5 min GPU). Arms: (1) fp32 baseline (61.55 raw / 62.28 DSL); (2) per-channel INT8
**all** convs; (3) INT8 **pointwise only**, depthwise fp16; (4) INT8 pointwise + dw7×7, the 40
dw3×3 mixers fp16. **This settles AC9 empirically, satisfies the two-protocol + paired-CI rule,
and cannot be misread the way the LSB argument was.** If arm 2 sits inside the paired band,
everything above is moot and you ship arm 2.

**Then the spatial-mixer histogram** (the still-unverified half of P2 — §8's tier-A5):
- **Which tensors:** the **40** matched by `endswith("layer_scale") and "token_mixer" in k`.
  The *other* 40 (block-level, `RepMixerBlock.forward:1075`) scale the **ConvFFN residual**, are
  **never folded into any kernel**, and are irrelevant here. `gate_checks.py:169` already
  implements this discriminator — **reuse it, don't rewrite it.**
- **Post-reparameterization, key on modules not names:** select modules having **both**
  `reparam_conv` and `reparam_conv_1d` (excludes stem/downsample/ConvFFN MobileOneBlocks, which
  also expose `reparam_conv`).
- **Two assertions that would have caught the 40-vs-80 error:** `n_modules == 40` and
  `Σ out_channels == 9920`.
- ⚠️ **Two shape traps** that would silently give a wrong number: the spatial kernel is
  `(C,1,3,3)` ⇒ reduce with `amax(dim=(1,2,3))`, **not** `(1,2)`; and `identity_kernel_like` must
  put the 1 at **`[1,1]`** for a 3×3, not at the 1-D centre index.
- **Headline quantity: `ρ_c = ‖ŵ_c − w_c‖₂ / ‖w_c − δ‖₂`** with `ŵ_c = s_c·round(w_c/s_c)`,
  `s_c = max|w_c|/127`. Three lines, no assumptions. **Run it on the temporal kernel too**, for a
  like-for-like number superseding the 36.2%.

**Panel's pre-registered prediction:** spatial λ ≈ 1.3e-2, so if |Δ_spatial| ≈ |Δ_temporal| ≈ 7.7
then λ|Δ| ≈ 0.10 ⇒ **n ≈ 12 LSB, b_eff ≈ 4.7 bits, ρ ≈ 0.02–0.05** ⇒ the *"spatial mixer runs at
1–3 LSB / ~2 effective bits"* claim is predicted **wrong in the conservative direction**.
Decision table: **ρ_median < 0.10** ⇒ spatial quantizes cleanly, **bury AC9's reparam-specific
rationale**, publish the one-paragraph mechanism note; **0.10–0.35** ⇒ AC9 collapses to a
one-line deployment rule (keep the 40 dw3×3 in fp16 = 0.59% of MACs), no QAT; **≥0.35** ⇒ AC9
gets its new rationale (sub-2-bit branch coding), fix still fp16-on-depthwise, still not QAT.

*Bonus 10-minute check that settles "should we unfuse" empirically: one forward pass recording
`‖mixer(x) − norm(x)‖ / ‖mixer(x)‖` per stage. If ≪1, the cancellation argument in §10.3 becomes
measured rather than estimated.*

### §10.5 🔺 The highest-value on-device number is available **today**, from graphs already exported

Activation traffic ≈ **86.4 MB/frame** (independently re-derived: 5.24·4 + 2.62·12 + 1.31·24 +
0.655·4 = 86.4 ✓ against §8.2) ⇒ **~693 MB at T=8**. Vision weights = 36M = **72 MB fp16**.

- **If Core ML amortizes weights across the T=8 frame batch:** total ≈ 765 MB ⇒ weight-only INT8
  saves 36 MB = **4.7%** ⇒ weight quantization is a **size** lever, not a speed one.
- **If the runtime re-reads weights per frame:** 693 + 576 = 1269 MB ⇒ INT8 saves **23%**.

> **Measure it in 30 minutes with what you already have:** you have Core ML exports at
> T ∈ {2,4,8,16}. **Plot latency vs T. Exactly linear ⇒ weights are not amortized. Sublinear with
> a positive intercept ⇒ they are.** This single unmeasured fact swings the quantization win from
> ~5% to ~23%, and it answers the `dynamic_axes` question for free.

**Predicted latency split (estimates, flagged as such):** **stage 1 is the surprise cost centre**
— 8.2% of MACs but 24.2% of bytes at intensity 60 vs ANE ridge 230 ⇒ ~26% of peak ⇒ expect
**20–30% of wall-clock, ~3× its MAC share**. **Stage 4 will be cheap** (≤8%, the only
compute-bound stage). **The 340 layout ops will cost more than their 21.5% share** on a
fixed-layout engine — expect **25–40%**, so P1's removal of 440 nodes / 80 full-tensor transposes
should buy **20–35% on ANE**, more than its 18.1% CPU figure. **GELU/BN are not free** (~5–15%,
pure bandwidth over 693 MB).

**On "1 checkpoint → N graphs":** **Core ML — essentially a non-issue if you use
`ct.EnumeratedShapes`** (pre-compiled per shape, stays ANE-eligible; one `.mlpackage`, **one
weight blob**, N shape specializations). ⚠️ **`RangeDim` / ONNX `dynamic_axes` is the wrong path
— it pushes the model off the ANE.** **TFLite/Android — real**: NNAPI/QNN recompile on resize and
often fall back to CPU, so ship static shapes; given `litert-torch` is blocked, **scope the paper
claim to Core ML and say so.** Structural fix, post-P1: 40 of 44 blocks become pure per-frame ⇒
split into a per-frame image graph with enumerated *batch* (~97% of weights) + a small stage-4/head
graph per T ⇒ N × (small graph). **Another item where P1 does the work.**

### §10.6 ✅ Indexing vs query — the open question §8.3 raised, now answered with a number

Text tower ≈ **1.22 GMACs/query** vs vision **63.97 GMACs/video** at T=8 ⇒ **the text tower is
1.9% of the vision tower.** If the product were query-latency-bound, the entire 86.5% analysis
would be optimizing the 98% that runs offline.

**Verdict: the paper's latency table is an indexing-throughput number, it should stay that way,
and the paper must say so in one sentence.** The deployable product for an on-device video-CLIP is
**indexing the user's own library** (the Photos-search shape) — that is where battery, thermals,
and "it took 40 minutes to index my library" live, and it is why MobileCLIP/MobileViCLIP report
*encoder* latency.

**Consequences that reorder §8:** **P11 + round4-P7 stay the main efficiency lane — confirmed.**
**P13 is a *size* item, not a latency item** (0 MACs) and must not be argued as an efficiency
result. **P8 stays demoted.** **But report the query path once anyway** — the two numbers are
~50× apart and a reviewer will ask.

### §10.7 What the plan is missing

- 🔺 **The gallery index dwarfs the model, and P6 makes it 8× worse.** 512-d fp16 = **1 KB/video**
  ⇒ 1M videos = **1 GB** vs a 72 MB model (**14–28×**). **round4-P6 (MaxSim) needs T=8 embeddings
  per video ⇒ 8 KB/video ⇒ 8 GB at 1M.** The plan prices P6 at "~4 GPU-h to test" and **does not
  price its 8× index cost at all — that belongs in P6's kill criteria.**
  Conversely **ANN *search compute* is a non-issue**: brute-force 512-d over 100k vectors = 51.2
  MMACs = **0.08% of one video encode** ⇒ use flat/IVF, **not HNSW** (whose graph costs
  1.1–1.5× vector memory to accelerate something that isn't the bottleneck).
- 🔺 **Matryoshka is a scheduling dependency nobody flagged.** 512 → 64/128-d truncation gives
  **4–8× index reduction** — but it is a **training-side loss change**, so it must be decided
  **before** the 6.6-day run, not bolted on after. P10 is currently parked as "a bonus loop inside
  C1"; that parking has a deadline.
- 🔺 **Video decode is not in the efficiency budget at all.** If the claim is indexing throughput,
  end-to-end per-video includes H.264/HEVC decode + sampling + resize + normalize ≈ **5–20 ms**,
  against an encoder plausibly at 30–60 ms ⇒ **decode is 10–40% of end-to-end**, and a reviewer
  will note that a 20% model speedup is a 12% product speedup. Also: put resize/normalize *inside*
  the Core ML graph via an `ImageType` input so the ANE/GPU does it, not the CPU.
- **Weight-only vs activation quantization:** on ANE fp16 is native and INT8 weights are
  decompressed on the fly unless you take the iOS 17+ W8A8 path. Activations are ~10× weights ⇒
  **weight-only INT8 ≈ 5% latency / 2× size. If you want latency you need *activation*
  quantization**, where the identity-tap analysis is irrelevant and ordinary calibration/outlier
  problems take over. The plan discusses only weights.
- **Free:** state that the text tower is frozen and inherited, so nobody reads it as this
  project's contribution being slow.

### §10.8 Benchmarking protocol (for C4's table)

**Devices:** **iPhone 12 Pro Max** is non-negotiable — it is the reference device for
MobileOne/FastViT/MobileCLIP and the only way to put your numbers on the same axis as the
baselines you cite; plus one current iPhone; plus one Android flagship **only if** TFLite actually
works, else omit and scope the claim.
**Protocol:** fixed OS/coremltools/Xcode versions **reported**; airplane mode, ≥80% battery,
unplugged; **50 warmup discarded, ≥1000 timed, report median + [p10,p90]** (matches what
FastViT/MobileCLIP report — a free reviewer defence); **compute-unit setting stated** and the
**actual dispatch reported** via `MLComputePlan` ("% ops on ANE" belongs in the table); 5-min
cooldown between models, **randomized model order**, plus a **3-min sustained-throughput** number;
load/compile time reported separately and excluded; **state T, resolution, batch, and whether
latency is per-video or per-frame.**
**What reviewers attack, most lethal first:** ① **competitor numbers copied from their papers**
(fatal, and the most common) — **measure every comparison yourself, same device, same day**;
② compute unit / ANE-fraction unstated; ③ **precision mismatch** between an fp32 accuracy table
and an INT8 latency table; ④ no thermal/sustained number; ⑤ mean instead of median, no dispersion;
⑥ per-frame vs per-video ambiguity; ⑦ end-to-end excluded (§10.7's decode point).

> ⚠️ **Amendment to §3's standing rule.** "*No training run is authorized for an efficiency claim
> until the random-init variant has a measured on-device number*" is correct for **dense fp16 and
> dense INT8**, but has one carve-out: **Core ML palettization and sparsity kernels are
> weight-value-dependent** — random weights are neither clusterable nor sparse, so a random-init
> measurement **understates** the palettized/sparse speedup. **Any P13(a) claim needs real weights.**

### §10.9 Considered and rejected (quantization round)

| Candidate | Why it dies |
|---|---|
| **GPTQ / AWQ / SmoothQuant** | LLM weight-only methods for the memory-bound *decode* regime with huge GEMMs. No ANE/TFLite-NPU kernel path for grouped INT4 dequant, and this model is **activation**-bound |
| **INT4 weights on the vision tower** | Depthwise layers are the classic low-bit casualty and the win is size-only. **Keep INT4 for the embedding table (P13a)** — a pure lookup with no accumulation |
| **Deploy unfused** | §10.3: +14% nodes, +20–70% activation traffic, moves cancellation into the activation domain, destroys the reparameterization pillar — to protect a branch P1 deletes |
| **QARepVGG branch-scale constraints** | §10.3: wrong disease (activation-variance explosion from a **BN'd** identity vs RepMixer's exact un-BN'd δ) |
| **RepOptimizer / RepOpt-VGG** | The elegant answer in principle (deploys a plain convnet) but needs a full retrain **plus** a hyper-search for branch scales, and RepMixer is not the constant-scale BN-free form RepOpt handles. 6.6 days for an unmeasured problem |
| **Cross-Layer Equalization (Nagel 2019)** | Recovers what per-channel weight quantization already gives you; coremltools and TFLite both do per-channel. Redundant |
| **Per-tensor weight scales** | Strictly worse, no upside |
| **LSQ/PACT QAT now** | Violates the standing rule twice: no on-device number exists, and no end-to-end evidence PTQ loses |
| **AdaRound / BRECQ / QDrop** | **Not rejected — held in reserve.** The correct escalation *only if* §10.4's 4-arm sweep shows plain PTQ losing >1.0 r_mean in both protocols |
| **`RangeDim` / ONNX `dynamic_axes` for elastic T** | Pushes the model off the ANE. Use `ct.EnumeratedShapes` |
| **HNSW for the on-device gallery** | Accelerates a search costing 0.08% of one encode, at +1.1–1.5× vector memory — the one scarce resource |

**Provenance (§10).** Quantization/deployment panel, resumed 2026-07-31. **Re-verified directly
against this worktree:** the LSB re-reading (§10.0, arithmetic), the 40-blocks/9920-channels
identity proving P1 subsumes P3 (§10.1), `norm`/`norm_1d` being pure-BN via
`use_scale_branch=False, num_conv_branches=0` at `:659-686` (§10.2), and the 86.4 MB/frame
traffic re-derivation (§10.5). **Not verified here:** every latency prediction and percentage
split, the ANE dilation call, the index-size and decode estimates, and all cited paper numbers —
**including two arXiv ids the panel itself flagged as unchecked.**

---

## §11. Round 5, MFU panel (2026-07-31) — **the last of the three; round 5 is now complete**

**It corrects §8.3's P12 ladder in three rungs and retracts one row of §8.4 (both mine).**

### §11.0 The roofline was wrong in a way that *shrinks* the prize — and that is the useful part

§8.3 estimated ~96 GB/s ≈ 2.9% of HBM. **Corrected by a dispatch trace run in this worktree:**
one `RepMixerBlock` forward issues **55 aten calls, 33 of them touching a full-activation tensor,
moving ≈92 N** (N = activation numel) — because at *train* time `MobileOneBlock` is 3-branch, so a
block carries **9 BatchNorms and 7 convs** (verified module counts: `BatchNorm2d:5, Conv2d:5,
BatchNorm1d:4, Conv1d:2`). Summing over the real geometry: **~511 GB/iteration/GPU forward**, ×4
for fwd+recompute+bwd ⇒ **~2.04 TB/iteration/GPU = 222 GB/s = 6.6% of HBM3** — the earlier
estimate was low by **2.3×** (its ×2.5 fudge over ConvFFN never accounted for those 9 BNs, 4 extra
convs and ~20 elementwise ops).

**The reframing matters more than the correction.** Per sample that is **31.9 GB** against 477
GFLOP ⇒ **the training step runs at ~15 FLOP/byte**, not the 60–240 of §8.2 — because **§8.2's
intensity table describes the *reparameterized inference* graph**, and the thing that trains is
4–16× more memory-bound. Against an H100 ridge of ~295:

> **MFU ceiling for this implementation = 15/295 = 5.1%. Measured 0.67% is 13.3% of its own
> roofline ceiling ⇒ real headroom is ~7.5×, not ~150×.**

This splits the win cleanly: **Axis A, kernel efficiency** (7.5× to reach HBM peak) and
**Axis B, bytes** (another 4–16× by raising intensity toward the inference graph — fusion,
deleting the temporal branch, `channels_last`, less recompute).

### §11.1 🔴 The missing suspect is the one §8.4 dismissed: **the temporal branch is ~33% of training traffic for 0.20% of MACs**

Verified structure at `:739-745`: the `reshape → permute(1,3,4,2,0) → reshape` chain
materializes as **two `aten::clone` calls per block** (measured, with source strides captured).
At B=64/T=8/stage-1 the first clone's innermost target axis is `t`, whose **source stride is
B·C·H·W = 41.9 MB** — a warp of 32 lanes reads 32 addresses 41.9 MB apart ⇒ **up to 16× DRAM
read amplification plus TLB thrash over a 335 MB working set.**

Everything between the clones then runs on shape `(B·H·W, C, T) = (262144, 80, 8)`, strides
`(640, 8, 1)` — **measured** — which is pathological twice over: the **contiguous run is T = 8
halves = 16 bytes** (every access wastes ≥50% of a 32-byte sector), and BN1d launches **one block
per channel = 80 blocks on 132 SMs** with only 8 of 32 lanes active.

**Accounting: ≈32 N of 96 N = 33% of all training memory traffic, and 15 of 33 full-activation
kernels per block. Estimated 50–70% of the 9.18 s iteration ⇒ removing it is worth 1.8–3×** —
larger than every rung of the original ladder combined, and **round4-P1 already deletes it for
export reasons.** (Why §8.4's 0.4% was wrong: see the retracted row there.)

> 🆕 **P15 · batch-offset temporal mixing — exact, zero layout ops, ~15 lines.** If the branch must
> survive (P4's controlled experiment needs it), it needs **no reshape at all**: frames sit at
> batch positions `t·B + b`, so a depthwise Conv1d over `t` is three scaled adds with a batch
> offset on the **contiguous NCHW** tensor. **Verified in float64 in this worktree:
> `max|batch-offset − reshape/permute/conv1d| = 4.44e-16`.** And BN1d over `(B·H·W, C, T)` reduces
> over **exactly the same element set per channel** as BN2d over `(T·B, C, H, W)`, so `norm_1d`
> becomes an ordinary NCHW BN2d. The branch goes from **≈32 N of mostly-uncoalesced traffic to
> ≈6 N fully-coalesced**, exact on current weights, no retraining. **Strictly better than P3**
> (filmstrip), which still needs one clone *and* produces the large-dilation depthwise conv2d
> §10.1 flags as an ANE-fast-path risk. **P15 has no layout op at all.**

### §11.2 The corrected P12 ladder

| rung | §8.3 claimed | **corrected** | evidence |
|---|---|---|---|
| `channels_last` | 2–4× | **1.2–1.5×**, and it is rung **4** | NVIDIA's 2–4× is measured on ≥90% **k×k dense** convnets. Here 86.5% is **1×1**, which maps *directly* onto a GEMM with **no layout transform** (1.1–1.4×); the real wins are depthwise (3–8×, but 3.9% of MACs) and **BatchNorm (1.5–3×, which the ladder never mentioned)**. Time-weighted ⇒ ~1.3× |
| `cudnn.benchmark` | 10–30% | 🔴 **0% — already `True`** | `train_media_types` = sorted **set** over corpora (`dataset/__init__.py:172`); `internvid_all` expands to v1–v11, **all `media_type="video"`** ⇒ set = `{"video"}` ⇒ len 1 ⇒ `True`. Only fires if an image corpus is added (e.g. P8's replay). Keep as an unconditional guard; book nothing |
| fp16 loss scaling | 5–50% | 🔴 **0%** | `config.py:140`: 「2026-07-22 实测:scale=32 全程**零** OVERFLOW」. **And bf16 would not help**: ATen's cuDNN-BN predicate excludes bf16 explicitly |
| `find_unused_parameters` | 10–40% | 🔴 **0% — dead code** | DDP is constructed **only in the non-DeepSpeed `else` branch** (`shared_utils.py:72-83`); with `deepspeed.enable=True` control goes to `:137`, `model = model_without_ddp`, `deepspeed.initialize`. **DDP is never instantiated.** Delete the flag as hygiene |
| selective checkpointing | +18% | **+8–10%** | +18% priced recompute as *FLOPs*; at 15 FLOP/byte it costs a full extra pass of **traffic** (511 GB of 2.04 TB = 25%). Skipping stages 3+4 removes 40% of that ⇒ ~10%. Sequence **after** §11.1, which changes forward traffic |

> 🆕 **New rung, and it outranks four of the five above: `model.half()` disables cuDNN
> BatchNorm for ~396 BN layers.** Verified: `shared_utils.py:66` does `model = model.half()`,
> casting BN `weight`/`bias`/`running_*` to fp16; `pretrain.py:112` then enters
> `autocast(float16)`. **This is not mixed precision — it is pure fp16 with fp16 norm params.**
> *(Inferred, and M5 settles it in 20 lines:)* ATen's cuDNN-BN predicate requires
> `input != kHalf || weight == kFloat`; fp16 input **and** fp16 weight makes it false ⇒ dispatch
> falls to `native_batch_norm` (two full passes, grid = C blocks) instead of cuDNN's fused
> persistent BN. Those BNs carry **~20% of forward traffic**. **Fix is two lines** — `.half()`
> then `.float()` the norm modules, standard AMP practice — **estimated 10–20%**, with the
> side benefit of removing fp16 `running_var` accumulation, a real numerical hazard for a model
> whose eval path depends on running stats.

**Also checked and closed:** **SyncBatchNorm is absent** (zero grep hits) — cost 0, and **do not
add it**: 396 layers × 2 (recompute) all-reduces per step would plausibly *be* a 2× regression.
**ZeRO-1 comm ≤ 0.2%** (≈36–40 M trainable ⇒ ~72 MB all-reduce) — not a suspect at this size.
**Host/eager overhead ~0.1–0.5%** (~7,000 kernels/iteration at 9.18 s = 1.3 ms per kernel, three
orders above dispatch cost), though `pretrain.py:140,143` force a device sync every iteration and
`log_freq=1` formats a line every step — fix if free. **Wave quantization is a non-issue for the
convs** (stage-1 GEMM N = 2.097 M ⇒ 16,384 tiles) — but **K = 80…640 is small**, so a fp16
tensor-core GEMM has a 1.25-deep k-loop and typically reaches only **20–35% of peak**. That is
architectural; budget it as residual, not as a rung.

### §11.3 🔴 Correctness bug (not MFU): checkpointing **double-updates every BatchNorm's running stats**

**Independently reproduced here.** One step, `nn.BatchNorm2d`, `use_reentrant=False`:

```
no-ckpt : num_batches_tracked=1   running_mean[:2]=[-0.002902, 0.004248]
w/ ckpt : num_batches_tracked=2   running_mean[:2]=[-0.005514, 0.008071]
```

The backward recompute re-runs BN in training mode, applying the EMA a **second time on the same
batch**. With `gradient_checkpointing=True` (default) and ~396 BN layers, **every checkpoint this
repo has trained has BN running stats with effective momentum ≈0.19 instead of 0.10.**

⚠️ **Note the irony to record:** `use_reentrant=False` was added *deliberately* (#316) to fix the
DDP `marked as ready twice` crash. **One fix introduced the other bug.**

Accuracy impact is probably small (faster EMA, same fixed point), but it is systematic and it
**interacts directly with P11's mandatory BN-recalibration control** — the recalibration pass
**must run with `with_cp=False`**, or it inherits the same doubling.

### §11.4 Measurement plan and the corrected action ladder

**Run M0+M1+M2 first — one profiler script, ~40 minutes, and it decides the whole ladder.**
- **M0** `torch.profiler(CUDA)`, 20 steady steps after 10 warmup ⇒ **Σ(kernel durations)/wall**.
  **< 0.85 ⇒ host stall; ≥ 0.85 ⇒ bad kernels.**
- **M1** same trace, `key_averages()` top-30. **`clone`/`copy_`/`batch_norm*` in the top 5 ⇒
  §11.1+§11.2's new rung confirmed**; ≥60% in gemm/conv ⇒ the model is genuinely GEMM-bound.
- **M2 (highest value)** baseline vs `layer_scale_1d = 0` **and** the `:739-745` reshape chain
  bypassed. **Δ ≥ 40% ⇒ §11.1 confirmed. Δ < 10% ⇒ §8.4 was right after all and §11.1 drops.**
- Then as needed: **M3** `ncu` sectors/request on the clones (≥8 of 32 ⇒ the 16× amplification is
  real); **M4** `dram__bytes` vs the 2.04 TB model; **M5** the 20-line cuDNN-BN fp16-vs-fp32-weight
  timing; **M6** `num_alloc_retries`; **M9** an isolated 1×1 conv to find the hard GEMM ceiling.

**Corrected ladder:** ① **P1 delete the temporal branch** (already decided) **or P15** if P4 needs
it alive — **1.8–3×** · ② **BN params in fp32** (2 lines) — **1.10–1.25×** · ③ **`channels_last`**
(must follow ①) — **1.2–1.5×** · ④ selective checkpointing — **1.08–1.12×** · ⑤ `torch.compile`
/ manual BN+GELU+residual fusion — 1.2–1.6× after ①–③ · ⑥ hygiene (unconditional
`cudnn.benchmark`, delete `find_unused_parameters`, raise `log_freq`, drop the per-step
`.item()`s) — ~1.0× · ⑦ fix §11.3 (correctness, not speed).

> **Compounded, realistic: 3–6× ⇒ 9.18 s/it → 1.5–3.0 s/it ⇒ 6.6 days → 1.1–2.2 days per
> 3-epoch run.** That is the single largest change to this project's planning constant.

**Stopping rule — when 6.6 days is physics.** **Tier 1:** if M0 busy > 0.90 *and* M2 Δ < 10%,
§11.1 is wrong and only kernel quality remains. **Tier 2:** run M9 — if an isolated stage-1 1×1
conv in fp16/`channels_last` with no framework overhead achieves **< 60 TFLOP/s**, then ~6% MFU is
the machine's answer for this op shape; combined with §11.0's 5.1% ceiling the honest target is
**~3–5% MFU ⇒ ~5–7× total**, and anything beyond needs changing the **model** (round4-P7
resolution, P11 depth), not the code. **Tier 3:** if rungs ①–③ together deliver **< 1.5×**
measured, **stop**, write the negative down with §11.0's arithmetic, restore 6.6 days as a hard
planning constant, and redirect into P11 × round4-P7 — which cut the *work* rather than the
*overhead*.

**Provenance (§11).** MFU panel, resumed 2026-07-31, run on a CPU-only node — so everything is
read from code, executed on CPU here, or explicitly marked inferred. **Re-verified in this
worktree before landing:** `model.half()` at `shared_utils.py:66`; DDP never constructed under
DeepSpeed (`:72-83` vs `:137-149`); `train_media_types == ["video"]` ⇒ `cudnn.benchmark` already
`True`; the zero-overflow note at `config.py:140`; and **§11.3's BN double-update, reproduced
independently**. **Panel-measured on CPU (not re-run by me):** the 55-call/33-op/92 N dispatch
trace, the module counts, the clone strides, and P15's 4.44e-16 float64 equivalence.
**Inferred, flagged by the panel itself:** the cuDNN-BN dispatch predicate (M5 settles it), the
BN kernel grid/thread geometry, the 16× coalescing amplification (M3 settles it), backward ≈ 2×
forward, and **every per-rung speedup magnitude.**
