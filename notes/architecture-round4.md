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
made of, and they gate whether T=16 is trainable at all. torch 1.13.1 has no SDPA and the
sm_70 pin is load-bearing, so the zero-risk fix is **manual query-block chunking inside
`MHSA.forward`**, not a torch upgrade. Either fix it or delete the flags — right now they read
like a feature and are a trap.

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
