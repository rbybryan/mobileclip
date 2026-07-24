# Neural-Inspired Round 2 — four theorems, one gate, three proposals

**Date:** 2026-07-24

**Scope.** Round 1 (`notes/neural-inspired-directions.md`, `notes/neural-inspired-implementation.md`)
ranked eight brain-inspired ideas N1–N8 and turned three into buildable plans (A1 surprise
pooling, A2 graded windows, B1 geometry-KD). This note is the **second, deeper pass**: it
reports four *proved and numerically verified* structural facts about this architecture that
reorganize the whole problem, a **zero-training gate** that must run before any further
temporal work, and three genuinely new proposals.

It was produced by four independent expert reviews (cognitive neuroscience; population
dynamics; video-language prior art; DL training feasibility) plus a code-verification pass.
**The four reviews converged, independently, on the same three conclusions** — that
convergence is the reason this note supersedes parts of round 1 rather than merely extending it.

> ⚠️ **This note changes round-1 conclusions.** See §6. In particular, one already-merged
> plan (A1 surprise pooling) is weaker than round 1 claimed, and one class of ideas that
> round 1 treated as promising (trajectory/temporal auxiliary losses) is now argued to be
> **first-order incapable** of moving retrieval. Read §0 before acting on round 1.

---

## 0. Four structural facts (proved, then numerically verified)

All four are properties of *this* model, not general claims. Each was checked numerically at
machine precision (`~1e-15`); the checks are reproducible from the snippets below.

Let `z_1..z_T ∈ R^512` be the per-frame embeddings (T=8), `v = mean_t z_t` the video embedding
(`FastViT.forward:1377-1384`, `models/mobileviclip_small.py`).

### F1 — The head is **unconditionally** order-blind

`v` is the mean of a *set*. A mean is invariant under **every** permutation, for any
trajectory — straight, curved, or arbitrary. Verified: `‖mean(Z) − mean(reverse(Z))‖ = 3.8e-15`.

**Consequence.** All of this model's temporal-order sensitivity must be manufactured *inside
the backbone* and smuggled into **every** frame embedding as a clip-global signal — because
only what is shared across all frames survives averaging. The two candidate channels are the
depthwise temporal `Conv1d(k=3)` in stages 1–3 and the stage-4
`temporal_embedding` (`:1096`, applied at `:1105`). F2 shows the second one cannot work.

### F2 — **Additive** positional codes are exactly annihilated; only **multiplicative** binding survives

Decompose `z_t = c_t + p_t` (content + additive positional code). Then
`mean(z) = mean(c) + mean(p)`, and `mean(p)` is a constant that does not depend on **which
content sat at which position**. Permuting content across positions leaves `v` identical.

Verified: additive tagging → `1.9e-15` change under content permutation *and* under reversal.
Multiplicative binding (rotate 2-d planes by a frame-index-dependent angle) → change of `11.0`.

**Consequence, and it is directly about our code.** Stage 4 applies its temporal code
**additively**: `x = x + resize_temporal_embedding(self.temporal_embedding, t)` (`mobileviclip_small.py:1105`). By F2 its
contribution to the pooled embedding's order-sensitivity is **exactly zero** except through
downstream nonlinearity — an indirect, unmeasured channel. This is a sharper statement of
bottleneck (c) than round 1 had: the model has **no explicit conjunctive what×when channel**.

### F3 — The mean annihilates the trajectory's **first-order term**, and recovering it is free

Write `z_t = s + (t−t̄)·u + r_t`. Since `Σ_t (t−t̄) = 0`:

```
mean_t z_t  =  s + r̄          ← the velocity u contributes exactly nothing
```

Verified: `‖mean(Z) − s‖ = 2.8e-15` when `r=0`. The least-squares estimate of `u` is a **fixed
weighted sum with constant weights**:

```
û = Σ_t (t−t̄)·z_t / Σ_t (t−t̄)²      with Σ_t (t−t̄)² = T(T²−1)/12   (= 42 at T=8)
```

**And the mandatory T1 disclosure — it is an exact identity, not an analogy.** Approximate
rank pooling (Bilen et al., *CVPR* 2016) uses `α_t = 2t−T−1`; for T=8, `α_t = 84·w_t`.
Verified exactly. **The least-squares trajectory slope *is* approximate rank pooling, up to
scale.** Any proposal here must disclose this up front rather than let a reviewer find it.

### F4 — The retrieval loss is **exactly flat** in the frame-residual subspace

Because the loss depends on `{z_t}` only through `v`:

```
∂L/∂z_t = (1/T)·Jᵀ·(∂L/∂v)          ← identical for every t
⇒  for any zero-sum perturbation Σ_t δ_t = 0:   Σ_t ⟨∂L/∂z_t, δ_t⟩ = ⟨g, Σ_t δ_t⟩ = 0
```

**Two consequences, both severe and both cutting against round 1's instincts.**

1. **"The contrastive loss prevents collapse" is false here.** It pins the *mean*. Frame-residual
   collapse (`z_t → z̄` for all t) costs the main loss **exactly nothing** — it is not a risk,
   it is a free direction. Any slowness/straightening-flavoured auxiliary has collapse as its
   *global optimum* with nothing pushing back.
2. **Auxiliary losses defined purely on residual structure are first-order orthogonal to
   retrieval.** Straightening, temporal decorrelation, and latent next-step prediction all live
   in this null space.

There is also a cheap architectural path to the degenerate solution: a depthwise temporal
kernel `[a,b,c]` with DC gain `a+b+c = 1` is a pure temporal low-pass that **attenuates the
residuals while exactly preserving the temporal mean**. A few dozen kernel weights get you
there. That is why the audit in §2 measures kernel DC/AC ratio.

**Important scope limit (do not over-apply F4).** F4 kills *losses on the residual subspace*.
It does **not** kill architecture changes that alter the pooled vector itself — a reweighted
or augmented readout changes `v` directly and is fully visible to the loss. A1 (surprise
pooling), A2 (graded windows) and R2 below are all in the visible class.

---

## 1. The measurement crisis — our benchmarks cannot see temporal structure

This is the second finding all four reviews reached independently, and it gates everything.

- **A single frame is enough.** A single-frame-trained model with frame ensembling matches or
  beats multi-frame models on MSR-VTT / DiDeMo / ActivityNet retrieval (Lei, Berg & Bansal,
  *ACL* 2023, "Revealing Single Frame Bias"); the atemporal probe of Buch et al. (*CVPR* 2022)
  is competitive across the standard suite.
- **The number that should settle it: BLIP — an image-only model — reportedly reaches ~43.3
  zero-shot R@1 on MSR-VTT, above our locked baseline of 42.50.** Our headline benchmark is
  currently won by a model with no temporal modelling at all.
- **~10% of MSR-VTT test videos are temporally relevant** (RTime, *Reversed in Time*, ACM MM
  2024). Frame shuffling contributes near-zero across many models.
- **Frozen video-CLIP encoders sit at chance on time direction.** Reported probes put
  frame-centric encoders at 49.5–50.7%, and InternVideo2 **drops to ~49.5% after CLIP-style
  language alignment** — i.e. VTC alignment *erodes* temporal direction information.

**Two consequences.**

1. **Any temporal fix evaluated only on MSR-VTT/DiDeMo/ANet will measure zero by construction**,
   and we would wrongly conclude the idea failed. Order-sensitive evals are mandatory:
   **RTime (esp. RTime-Binary)**, SSv2-template/label retrieval, Vinoground, TemporalBench.
   Keep SSv2-MC but stop treating it as the temporal evidence — multiple choice over 174
   templates leaks appearance cues.
2. **RTime-Binary has an analytically known null.** If the pooled embedding is invariant under
   time reversal, a video and its reverse map to the *identical* vector and the ranking is a
   coin flip — **exactly 50.0%**, not merely "poor". That converts a vague question into a
   binary with a known floor, and it is the cleanest falsifier we have.

### 1.1 The decision threshold is probably below our noise floor

MSR-VTT 1k-A has n=1000 queries, so binomial SE on R@1≈45% is ≈1.6 points. Paired A/B cancels
much of that, but **training-seed variance does not**: expect roughly ±0.5–1.0 on R@1,
compounding to **±1.5–3 on R-Sum**.

`notes/paper-plan.md:127` still carries a **"≥ +0.3 R-Sum"** keep-criterion. §227-229 of the
same file already tightened this (multi-benchmark, both protocols, second seed for <0.5 calls),
but the two statements are inconsistent and the tighter one is still likely under-powered.

**Action: run the locked baseline a second time at full scale with a different seed.** One run
converts every subsequent arm from anecdote into evidence. Until then, prefer a **multi-dataset
sign test** (MSR-VTT + DiDeMo + MSVD + ANet, both directions, consistent sign in ≥6 of 8) over
any single-number threshold — a consistent +0.4 across eight cells is far stronger evidence
than one +1.2 on MSR-VTT.

---

## 2. Gate-0 — the temporal-capacity audit (zero training, run before anything else)

All four reviews independently proposed this. It is a few hours of measurement that can kill
or redirect an entire class of work, and it produces a paper figure either way.

### 2a. Static audit — from the checkpoint alone, no GPU, no data

`experiments/temporal_capacity_audit.py` (new, self-tested) reads a checkpoint and reports:

- **`‖temporal_embedding‖`** and its per-position spread. It is **zeros-initialized**
  (`:1096`), so if training never moved it, stage 4 is permutation-equivariant over frames and
  — by F1 — contributes nothing. *(By F2 it contributes ~nothing even when nonzero, since it is
  additive. Both numbers are worth having.)*
- **Temporal-kernel antisymmetric fraction** `‖w − flip(w)‖ / (2‖w‖) ∈ [0,1]`. Time reversal
  maps `[a,b,c] → [c,b,a]`, so a **symmetric kernel is exactly reversal-blind**. This is the
  direct measure of arrow-of-time capacity: 0 = reversal-blind, 1 = pure difference operator.
- **Temporal-kernel DC/AC ratio** `|Σw| / Σ|w|`. →1 means pure low-pass, i.e. the F4 collapse
  shortcut has been taken and frame residuals are being erased.

```bash
python experiments/temporal_capacity_audit.py --self-test          # no checkpoint needed
python experiments/temporal_capacity_audit.py checkpoints/mobileviclip_small.pt
```

**Read the verdict this way:** antisymmetric fraction median < 0.05 **and** DC/AC median > 0.95
⇒ the trained model barely encodes time direction ⇒ predict RTime-Binary ≈ 50%.

⚠️ The script has been self-tested and run end-to-end on a **synthetic** checkpoint, but **not
against the real weights** (Claude has no access to them). Training-state checkpoints store the
temporal branch un-fused (`mixer_1d` / `norm_1d`); the fused inference kernel is
`id_tensor + layer_scale_1d·(mixer_1d − norm_1d)` (see `RepMixer.reparameterize`). If the default key does not
match, the script prints the candidate keys it found — pass `--kernel-key`, or reparameterize
and re-save first.

### 2b. Dynamic audit — eval-only perturbations on the locked B0 checkpoint

Run the existing eval configs with the frames perturbed at test time, on **both** MSR-VTT
(appearance) and SSv2-MC (temporal):

| Perturbation | What it measures |
|---|---|
| **shuffle** (random permutation of the 8 frames) | total order-sensitivity of the deployed model |
| **reverse** (time-reversed frames) | arrow-of-time sensitivity specifically |
| **freeze** (middle frame repeated ×8) | how much *any* multi-frame information is worth |
| **T=16 at eval only** | possible free gain, and calibrates 1.6's multi-T curve |

Also record the **pooled-embedding cosine** between perturbed and clean video embeddings, at
stage-3 output, stage-4 output, and post-pool. That localizes *where* order is created or
destroyed — if stage-4 outputs are already shuffle-invariant, no aggregator fix can help and
the problem is upstream.

### 2c. What each outcome licenses

| Outcome | Meaning | Action |
|---|---|---|
| shuffle Δ ≈ 0 on **both** MSR-VTT and SSv2-MC, static audit shows symmetric low-pass kernels | model is order-blind **and** the suite cannot see it | R2 becomes the *only* temporal proposal worth funding, and it must be measured on RTime; everything else temporal is dead |
| shuffle Δ ≈ 0 on MSR-VTT, **> 0 on SSv2-MC** | model has order info; the *benchmark* is the problem | proceed with R2, add RTime/Vinoground, report as a Pareto result |
| shuffle Δ large on MSR-VTT | our premise is wrong | re-derive; F1 says this should not happen via the head, so it would mean strong backbone-side order coding |

**This is also a paper figure regardless of outcome** — a per-benchmark measurability ceiling
measured on our own model, rather than cited from someone else's, in the same slot as ROADMAP
1.6's multi-T degradation curve.

---

## 3. The three proposals

Ranked by (expected gain × novelty) ÷ (risk + cost), under T1/T2 from round 1 §0.

| # | Proposal | Attacks | Inference cost | Novelty | Verdict |
|---|---|---|---|---|---|
| **R1** | **Exact frozen-text negative bank** | **(d)** | **zero** (train-time) | low as research, high as engineering | **Do first — highest expected value** |
| **R2** | **Order-recoverable readout** (mean → +slope → +phase binding) | **(c)** | ~0.1% MACs, static | medium-high | **Do, gated on §2** |
| **R3** | **Straightening as *measurement*, not objective** | — | zero | medium (negative result) | **Do — free figure, no arm** |

### R1 · Exact frozen-text negative bank — the one all four reviews picked

**The observation.** `TextTransformer.encode_text` ends with a **single trainable op** —
`token_emb @ self.projection_layer` (`text_encoder.py:225`). Everything before it is frozen by
`freeze_text=True` (`mobileviclip_small.py:74-80`). Therefore **a caption's pre-projection
feature is a constant for the entire training run.**

**Why that matters.** Every memory-bank paper spends its effort managing *staleness*: the
encoder drifts, so queued features go stale, so you need a momentum encoder. Here the encoder
**structurally cannot drift**. Cache the pre-projection features once, apply the *current*
projection each step, and the bank is **mathematically exact** — identical to what the text
tower would produce for those captions right now. Zero staleness, no momentum encoder, no queue
bookkeeping.

**Why it is worth doing.** SigLIP's batch-size sweep is the reference point: below ~16k batch,
contrastive quality is strongly batch-limited, saturating around 32k. **We are at global batch
512 — roughly 30–60× below saturation.** This is the largest unexploited lever in the setup,
and unlike everything in §0 it acts on the *primary* objective rather than in its null space.

Two freebies fall out of the same frozen-tower property:
- **Offline false-negative masking.** InternVid captions repeat ("a man is talking"). At 512 the
  false-negative rate is tolerable; at 65k it is not. Because the text space is static, the
  near-duplicate mask can be precomputed **once, offline**. This composes with the existing
  `caption_idx` / `get_mask(idx)` multi-positive machinery (`criterions.py:287`).
- **Mined hard negatives.** Precompute each caption's nearest neighbours in the frozen text
  space and inject them as guaranteed-hard negatives.

**Asymmetry — state it honestly.** This gives *text* negatives, strengthening the v2t direction.
The t2v direction (our headline metric) needs banked *video* embeddings, which **do** drift and
would need a real momentum queue. Start text-only because it is exact and free; add a modest
video FIFO only if t2v lags.

**Sketch** (default-off = bitwise-identical by construction; note the **early return**, not
`+ 0*aux` — multiplying by zero still executes ops and can change fp16 reduction order):

```python
def vtc_with_bank(v, t, bank_feats=None, text_proj=None, dup_mask=None, logit_scale=100.0):
    logits = logit_scale * v @ t.t()                        # [B, B]  baseline path
    if bank_feats is not None and bank_feats.numel() > 0:
        t_bank = F.normalize(bank_feats @ text_proj, dim=-1)  # exact: no staleness
        extra  = logit_scale * v @ t_bank.t()               # [B, K]
        extra  = extra.masked_fill(dup_mask, float('-inf'))  # precomputed offline
        logits = torch.cat([logits, extra], dim=1)
    return F.cross_entropy(logits.float(), torch.arange(v.size(0), device=v.device))
```

**⚠️ fp16 overflow trap.** With `temp = 1/100` the logit scale is 100, and a `512 × 65536`
logit matrix in fp16 will overflow before the pool is exhausted. Compute the logit matrix and
log-softmax in **fp32** with explicit max-subtraction, and chunk the matmul over the bank
dimension. Log `max|logit|` and the DeepSpeed skipped-step count. Logged to `TROUBLESHOOTING.md`.

**Non-neuro twin (T1).** MoCo (He et al., *CVPR* 2020), XBM (Wang et al., *CVPR* 2020), and
MEEL (arXiv:2103.15686) which already builds cross-modal memory banks for video-text retrieval.
The twin is strong and well-established. **The unclaimed part is the *exactness* argument**, and
the neuroscience framing (complementary learning systems: a stable consolidated store replayed
into a fast learner, valid precisely *because* the store does not drift) should be labelled as
what it is — a rationalisation of an engineering fact, not a derivation. Per T1, do not sell
this as brain-inspired; sell it as engineering and take the number.

**Kill criterion.** 512 → 65k negatives yields < 0.5 R@1 ⇒ (d) is misdiagnosed; stop investing.
**Instrumented early check:** log the fraction of softmax mass on banked vs in-batch negatives
(<5% ⇒ the bank is inert, kill the arm), and the rank of the positive among negatives — if the
positive is already rank-1 against 512 negatives for most samples, the task is saturated at this
batch size and we have the answer before running the sweep.

**Bundle in, do not spend separate arms on:** the **temperature** (fixed 1/100 was inherited
from image-CLIP and is on the sharp end; the optimum shifts with negative count — make it
learnable or sweep it), and the **sigmoid/SigLIP loss** — which is *already implemented*
(`model["use_sigmoid_loss"]`, `vtc_loss_sigmoid`, `experiments/configs/r1a_sigmoid.py`) with
learnable t/b and the correct SigLIP init. Sigmoid + large pool **is** the SigLIP regime.

### R2 · Order-recoverable readout — the structural fix for (c)

**The insight that unifies three independent recommendations.** The three reviews proposed three
different readout fixes; they are **one family at increasing order**, which the numerical check
confirms (the linear ramp correlates **0.994** with the lowest-frequency phase channel):

| Order | Readout | Order-sensitivity | Cost |
|---|---|---|---|
| 0 | `mean` (current) | **none** (F1) | — |
| 1 | `+ û` = fixed-weight slope ≡ **approximate rank pooling** (F3) | arrow of time | ~4k MACs |
| k | `+` phase binding: rotate 2-d planes by `ω_j·t` before summing | finer order structure | ~2k MACs |

All are **static, dense, data-independent** — one einsum plus a constant table, then one
1024→512 projection (~0.5M MACs, ≈0.1% of the vision tower). Core ML / TFLite exportable as-is.
This is the rare mechanism that is genuinely free under T2.

```python
# Z: [B, T, C] frame embeddings, surfaced from FastViT.forward (see §5 plumbing)
u0 = Z.mean(1)                                        # existing centroid
c  = torch.arange(T, dtype=Z.dtype) - (T - 1) / 2       # T-general (2.1 made T dynamic)
w  = c / (c * c).sum()                                  # == (t-4.5)/42 at T=8
u1 = torch.einsum('t,btd->bd', w, Z)                  # first-order term (F3)
v  = F.normalize(torch.cat([u0, gamma * u1], -1) @ W_proj, dim=-1)   # W_proj: 1024->512
```

`gamma` is a learnable scalar **init 0** and `W_proj` zero-init on the `u1` block, so **step 0
is exactly the current mean-pool** (C-switch convention). LayerNorm each branch before
concatenation — otherwise `‖u1‖ ≪ ‖u0‖` and after L2-normalisation the branch contributes
negligible angle.

**The decisive experiment, and it is the best thing in this note.** F2 predicts that **additive**
positional codes must fail and **multiplicative** binding must work. So run:

```
arm-mean      : baseline
arm-additive  : learned additive per-frame code before the mean   ← F2 predicts NO GAIN
arm-slope     : + rank-pool slope (F3)                            ← predicts gain
arm-phase     : + rotary phase binding                            ← predicts gain ≥ arm-slope
```

If `arm-additive` matches `arm-phase`, F2 is wrong and the upstream nonlinearity already does
the binding. If `arm-additive` is flat and the others move, the theory is confirmed on our own
model. **Either result is publishable and cheap**, and it is falsifiable in a way most
brain-inspired pooling papers are not.

**Non-neuro twins (T1), all mandatory baselines.** (i) **Approximate rank pooling / dynamic
images** — Bilen et al., *CVPR* 2016 / *TPAMI* 2018 — which F3 shows is the *exact identity*
for the order-1 term, so disclose it up front; (ii) additive learned positional codes;
(iii) **RoPE** (Su et al.) — same rotation algebra, but applied inside attention to
queries/keys, never as a *pooling-survival* mechanism; (iv) CLIP4Clip `seqTransf`, the
brute-force version, worth **+1.4 R@1 on MSR-VTT (43.1 → 44.5)** while `seqLSTM` was *worse*
than mean pooling — **that +1.4 is the realistic ceiling for order-aware aggregation on
MSR-VTT**, and `seqTransf` pays real inference cost. Calibrate expectations to a few tenths on
MSR-VTT and put the weight on RTime.

Neuroscience basis for the phase variant: theta-gamma phase coding (O'Keefe & Recce,
*Hippocampus* 1993; Lisman & Jensen, *Neuron* 2013 — successive items occupy successive gamma
cycles, so *order is phase*; Lisman & Idiart, *Science* 1995 derive a 7±2 capacity, close to our
T=8); temporal context drift (Howard & Kahana, *J. Math. Psych.* 2002); nonlinear mixed
selectivity as what makes variables decodable (Rigotti et al., *Nature* 2013). The formal
ancestor is Holographic Reduced Representations (Plate, *IEEE TNN* 1995).

**Kill criterion.** RTime-Binary stays at 50.0% ⇒ the readout carries no order information ⇒
premise wrong. Secondary: ablate `gamma → 0` at eval; if R@k is unchanged the branch is
decorative. Also kill if MSR-VTT regresses beyond a **pre-registered** no-regression margin —
set that margin *before* seeing numbers.

### R3 · Straightening as a measurement, not an objective

Round 1's instinct — and the obvious brain-inspired move — is a perceptual-straightening loss
(Hénaff, Goris & Simoncelli, *Nat. Neurosci.* 2019; Hénaff et al., *Nat. Commun.* 12:5982, 2021).
**Three independent reasons say do not make it an objective:**

1. **F4** — it lives in the null space of the retrieval gradient, and its global optimum
   (residual collapse) is free.
2. **Prior art is ~85% taken.** Straightening-as-a-training-objective: Niu, Savin & Simoncelli
   (*NeurIPS* 2024). Straightening in video, explicitly citing Hénaff: **LiFT / "Chirality in
   Action"** (Bagad & Zisserman, *NeurIPS* 2025). And the non-neuro twin is from **2016** —
   Jayaraman & Grauman's "Slow and Steady Feature Analysis" penalizes second-order temporal
   derivatives, which is straightening minus the neuroscience; the curvature loss differs only
   by being scale-invariant.
3. **It cancels with our head.** For an idealized straight trajectory `z_t = z_s + (t/T)z_d`,
   the mean is the **midpoint** `z_s + z_d/2`, which is *identical* for the reversed video.
   Straightening the trajectory and then averaging it is self-defeating if the goal is temporal
   competence. LiFT works only because it keeps `z_d` as a separate **signed** dynamic token —
   exactly what R2's order-1 term restores.

**What to do instead — free, and a genuine result.** *Measure* trajectory curvature on the
baseline and log it over training. Straightening is reported to *emerge* in robust networks
(Toosi & Issa, 2023) without being asked for. The curvature metric (Hénaff convention):

```python
d = Z[1:] - Z[:-1]; d = d / (norm(d, axis=-1, keepdims=True) + 1e-8)
curvature = degrees(arccos(clip((d[:-1]*d[1:]).sum(-1), -1, 1))).mean()
```

Report the **distribution** over clips, always against three nulls: **frame-shuffled** (primary
— is there any temporal structure at all?), **pixel-domain** (the Hénaff denominator; report
relative curvature `c_model − c_pixel`), and **untrained-backbone**. Compute on raw embeddings —
never on PCA/smoothed coordinates, since a low-dimensional projection of a random walk can look
like a beautiful curve. Curvature is **not comparable across sampling rates**, so fix T and
stride across all arms.

**The likely headline, and it is worth having:** at T=8 and ~1 fps our 8 points are plausibly a
tight random walk around a semantic centroid, not a smooth curve — in which case curvature sits
at the shuffled-null value (~90°) and *there is no trajectory to straighten*. A clean flat
dose-response between curvature and R@k is a **citable negative result** against a hypothesis
the field currently treats as self-evidently good, at zero A/B cost.

---

## 4. Killed in this round, with reasons

| Idea | Why killed |
|---|---|
| **Straightening / slowness / temporal-decorrelation as *losses*** | F4 (null space + free collapse); prior art taken; cancels with mean pooling. Keep only the **variance floor** term (`relu(γ − std_t)`) as a reusable collapse guard |
| **Latent next-step prediction (V-JEPA-as-auxiliary)** | F4; collapse reaches loss exactly 0 under an identity predictor; needs stop-grad **and** EMA **and** predictor asymmetry **and** a variance floor — four guards around a term with no mechanism; the cell is closing fast in the 2026 literature |
| **Event-segmentation pooling** | **T=8 has no room** — 7 gaps, segments of 2–4 frames, and at ~1 fps every gap is arguably already a boundary. Adds dynamic control flow (T2 violation). Revisit only if T ≥ 32 on long-video benchmarks |
| **Coarse-to-fine / temporal-blur curriculum** | Twin is *random* T∈{2,4,8,16} which C2 already does; curricula need schedule room and we have **3 epochs from a pretrained init**; FluxViT (*ICCV* 2025) owns the resolution-curriculum lane. Also a design trap: an ordered curriculum must match `E[T]` and the full marginal, or it measures "more frames = more compute" |
| **Temporal-order / shuffle / arrow-of-time prediction heads** | Superseded by masked video modelling; forces frames to encode phase, which F1 says the mean discards; fights the VTC objective, which is itself reported to *erode* temporal direction |
| **Temporal decorrelation across T** | Statistically impossible as posed: a 512×512 covariance from **T=8 samples** has rank ≤7 |
| **Any new token merging / pruning / selection** | Lane closed inside 18 months: TempMe (*ICLR* 2025), ResidualViT (*ICCV* 2025), EVS, FluxViT |
| **Frame-difference input front-end** | A dense conv costs the same regardless of input energy, so it cannot attack (a) — its only justification. General conclusion worth recording: **no teacher-free, train-time-only idea can attack (a)**; (a) requires an inference-time change |

**Two things demoted rather than killed** (do them, but not as brain-inspired contributions):
**factorized / divided space-time attention** for bottleneck (b) — cuts attention memory from
512² to 64×8², static and export-friendly, and probably the correct fix, but it is TimeSformer /
axial / Video-Swin and the neuroscience adds nothing; and a **small trainable text adapter**
(LoRA r=8–16 on the last text blocks; note `open_text_lora` already exists in the config
schema) — plausibly high-leverage since LiT caps how well the two spaces can meet, but it
changes the stated LiT design, so it is a bigger decision than a loss term.

---

## 5. Engineering prerequisites (verified against the code)

**P1 — Surface the per-frame trajectory.** `FastViT.forward` builds `[T,B,C]` at
`mobileviclip_small.py:1377` and returns only the pooled `[B,C]` at `:1384`; losses live one
level up in `MobileViCLIP_Small.forward` (`:126-155`), which never sees it. **Every** trajectory
proposal (R2's readout, R3's curvature logging, any residual diagnostic) needs this one-time
plumbing change. It is shared infrastructure: pay it once, and the marginal cost of the next
idea is near zero. Keep the default path bitwise-identical (return the extra tensor only when
requested).

**P2 — The `loss_weight` trap (affects already-merged plans).** `tasks_clip/pretrain.py` does
`loss = sum(loss_dict.values())`, and `criterion.loss_weight` is used **only** to build the
logging meters (`loss_names = [... if v != 0]`, `:48`). **The configured weight is never
multiplied into the loss.** A new auxiliary term must apply its own weight inside `forward`
(reading `self.config.criterion.loss_weight`), or setting `kd_geo=0.5` in a config will silently
change nothing but the log output. **This applies to the already-merged B1 geometry-KD plan.**

**P3 — Loss balancing, for any auxiliary term we do run.** Set λ by **gradient-norm matching**
at step 0 (`λ = ρ·‖g_main‖/‖g_aux‖`, `ρ ∈ [0.05, 0.1]`) rather than guessing — a guessed constant
does not transfer from the 10k pilot to 10M, which is a common way pilots mislead. Ramp λ from 0
over the first 5–10% of steps, **starting after** the LR warmup (two simultaneously moving
schedules produce unattributable instability). Beware the `different_lr` interaction: a new head
at 1e-3 against a 1e-5 backbone is a **100× ratio**; stop-gradient from the head into the trunk
for the first ~500 steps.

**P4 — The pilot diagnostic that saves full-scale arms.** The 10k pilot cannot rank methods, but
it *can* prove an idea is mechanically incapable of helping. Log the **gradient cosine between
the auxiliary and main losses** at the last shared block: persistent `cos ≈ 0` is the empirical
confirmation of F4 (the term is in the null space) ⇒ **kill the arm before spending an H100 run**.
`cos < 0` ⇒ direct conflict. `cos ∈ [0.05, 0.3]` ⇒ the only outcome justifying a full arm. Also
log `‖g_aux‖/‖g_main‖` (outside [0.01, 0.3] ⇒ λ miscalibrated) and, as the architecture-specific
collapse tell, the **depthwise temporal-kernel DC/AC ratio** trending to 1.

---

## 6. Corrections to round 1

1. **A1 (surprise pooling) is weaker than round 1 claimed.** It survives F4 (a reweighting
   changes the pooled mean, so it is visible to the loss) — but F1/F2 show that *any* weighting
   scheme, including the attention pool it must beat, remains **order-blind**: reweighting a set
   does not order it. A1 can only exploit *appearance* saliency (down-weighting blurry or
   uninformative frames), never temporal order. Round 1 framed it as the aggregator fix for the
   predictive-coding story; it is not. R2 is the order fix.
2. **N4 / A2 (graded temporal windows) is unaffected by F4** (it changes the features and hence
   the mean) but is capped by §1's measurability ceiling — its expected win on SSv2-MC should be
   re-read against the audit, not assumed.
3. **B1 (geometry-KD) inherits the P2 loss-weight trap** — its config sketch sets
   `criterion["loss_weight"].update(dict(kd_cos=1.0, kd_geo=1.0))`, which by P2 does nothing to
   the optimization. Fix when implementing.
4. **Round 1's "profile the per-frame vs stage-4 MAC split" prerequisite still stands** and is
   now joined by the §2 audit as a second zero-cost gate.

---

## 7. Sequencing — extends `experiments/ROADMAP.md` 第二线

```
GATE-0 (now, zero training — blocks all temporal work)
  2.6a  static audit:  python experiments/temporal_capacity_audit.py checkpoints/mobileviclip_small.pt
  2.6b  dynamic audit: shuffle / reverse / freeze / T=16 evals on B0, MSR-VTT + SSv2-MC
  2.6c  add RTime (esp. RTime-Binary) + SSv2-template to the eval harness
  2.6d  baseline seed replicate at full scale  -> the noise floor (§1.1)

THEN, in parallel:
  R1  exact frozen-text negative bank (+ temperature, + sigmoid already in r1a)
        no dependency on GATE-0; highest expected value; kill if <0.5 R@1 at 65k
  P1  surface the [B,T,C] trajectory   (unlocks R2 + R3; one-time plumbing)

GATED on 2.6:
  R2  order-recoverable readout, ablation ladder mean / additive / slope / phase
        primary metric RTime-Binary; MSR-VTT as pre-registered no-regression check
  R3  curvature logging + the three nulls  (free figure, no arm)
```

**Do not fund** a further arm on straightening-as-loss, latent prediction, event pooling, or
curriculum (§4).

---

## 8. Provenance and citation hygiene

**Verified by me, in this repo:** F1–F4 (numerically, at machine precision, plus the analytic
derivation of F4); the ARP identity `α_t = 84·w_t`; every code line reference cited above; the
audit script's metrics (self-tested, incl. a bug found and fixed — the antisymmetry metric
needed a factor of 2 to land in [0,1]).

**NOT verified by me:** all external empirical numbers (BLIP 43.3 on MSR-VTT; the ~10%
temporal-relevance figure; CLIP4Clip's 43.1→44.5; the arrow-of-time probe percentages; SigLIP's
16k/32k thresholds) and all bibliographic details. These came from four independent literature
reviews and several agreed with each other, but **venue/year/arXiv IDs must be checked before
they enter a submission** — several 2025–2026 arXiv IDs in particular were reported without a
retrieved link. Treat §1's numbers as *motivating* until the audit in §2 reproduces the effect
on our own model, which is precisely why §2 is the gate.

**References (anchors; verify at submission).** Perceptual straightening — Hénaff, Goris &
Simoncelli, *Nat. Neurosci.* 2019; Hénaff et al., *Nat. Commun.* 12:5982, 2021; straightening as
objective — Niu, Savin & Simoncelli, *NeurIPS* 2024; LiFT / Chirality in Action — Bagad &
Zisserman, *NeurIPS* 2025 (arXiv:2509.08502); non-neuro twin — Jayaraman & Grauman, *CVPR* 2016;
SFA — Wiskott & Sejnowski, *Neural Comput.* 2002; rank pooling / dynamic images — Bilen et al.,
*CVPR* 2016 / *TPAMI* 2018; phase coding — O'Keefe & Recce, *Hippocampus* 1993; Lisman & Idiart,
*Science* 1995; Lisman & Jensen, *Neuron* 2013; temporal context — Howard & Kahana, *J. Math.
Psych.* 2002; mixed selectivity — Rigotti et al., *Nature* 2013; HRR — Plate, *IEEE TNN* 1995;
CLS — McClelland, McNaughton & O'Reilly, *Psych. Rev.* 1995; event segmentation — Zacks et al.,
*Psych. Bull.* 2007; Baldassano et al., *Neuron* 2017; Franklin et al., *Psych. Rev.* 2020;
single-frame bias — Lei, Berg & Bansal, *ACL* 2023; ATP — Buch et al., *CVPR* 2022; RTime —
*ACM MM* 2024 (arXiv:2412.19178); Vinoground — arXiv:2410.02763; TemporalBench —
arXiv:2410.10818; SigLIP — Zhai et al., *ICCV* 2023; LiT — Zhai et al., *CVPR* 2022; MoCo — He
et al., *CVPR* 2020; XBM — Wang et al., *CVPR* 2020; MEEL — arXiv:2103.15686; X-Pool — Gorti et
al., *CVPR* 2022; CLIP4Clip — Luo et al. 2021; TempMe — *ICLR* 2025; ResidualViT — *ICCV* 2025;
FluxViT — *ICCV* 2025; VICReg — Bardes, Ponce & LeCun, *ICLR* 2022.

*Code references checked 2026-07-24 against `models/mobileviclip_small.py`,
`models/backbones/internvideo2/mobileclip/text_encoder.py`, `models/criterions.py`,
`tasks_clip/pretrain.py`.*
