# Neuroscience-Inspired Directions (code-grounded)

**Date:** 2026-07-23

**Scope.** This note answers a specific question: *are there **neuroscience-inspired**
mechanisms that would make MobileViCLIP faster on-device and/or more accurate, and
which of them survive contact with (a) this repo's actual code and (b) a mobile
deployment target?* It is a deliberate complement to the existing model notes,
which contain **no brain-inspired ideas**:

- `notes/model-architecture-improvements.md` — the P1–P10 engineering/objective backlog;
- `notes/paper-plan.md` — the C1 (Video Dataset Reinforcement) / C2 (temporal-elastic
  distillation) / C3 (components) CVPR spine;
- `notes/whitespace-directions.md` — data, temporal hard negatives, quantization, export.

Where an idea here maps onto an existing P#/C#, that is called out explicitly — the goal
is to slot brain-inspired mechanisms **into the existing plan**, not to open a parallel one.

**Implementation:** the executable, feasibility-ordered follow-through — exact touchpoints,
configs, tests, and kill criteria, sequenced into `experiments/ROADMAP.md` 第二线 — is in
`notes/neural-inspired-implementation.md`.

> ⚠️ **Partly superseded by [`notes/neural-inspired-round2.md`](neural-inspired-round2.md)
> (2026-07-24).** Round 2 proves four structural facts about this architecture that change
> several conclusions here: the mean-pool head is **unconditionally** order-blind (F1);
> **additive** positional codes — including our stage-4 `temporal_embedding` — are exactly
> annihilated by it (F2); the retrieval loss is **exactly flat** in the frame-residual
> subspace, so trajectory/slowness-style auxiliary losses are first-order incapable of
> moving retrieval and frame-collapse is a *free* direction (F4). It also argues our current
> benchmark suite cannot measure temporal ideas at all, and adds a mandatory **zero-training
> gate (GATE-0)** before any further temporal work. Read round 2 §0 and §6 first.
> **GATE-0 has since run and closed (2026-07-27, `experiments/gate0_results.md` §8):** three
> independent probes agree the official weights carry **no temporal-direction information**
> (shuffle ≈0, kernels indistinguishable from random, temporal gate at 6.1% of spatial), while
> **frame diversity is worth ≈13 R@1**. The order-sensitive benchmark suite was therefore
> **not bought** (⏸ 暂缓) and the budget went to round-2 **R1** instead. Read every "temporal
> order" motivation in this note against that result.

Anchored to code checked 2026-07-23 (`models/mobileviclip_tiny.py`, `_small.py` is the
same code at larger width; re-locate by symbol if lines drift).

---

## 0. Two governing truths (read before any proposal)

Two constraints dominate every idea below and matter more than any single principle.
Both were reached independently by a video-CLIP feasibility review and a
brain-DNN-alignment review; the convergence is the reason they lead this note.

**T1 — The real lever is temporal redundancy, not "brain-ness."** The dominant per-frame
cost (see §1) exists because the full spatial backbone runs ×T=8 over frames that
are near-identical at 1 fps. Every idea that actually buys latency — predictive-coding
gating, two-stream, adaptive-T — is a *different disguise for exploiting inter-frame
redundancy*, a well-understood efficient-video-recognition insight. **The neuroscience is
motivation, not mechanism.** The paper-level consequence: a neuro-motivated block must
beat the *non-neuro version of the same computational trick* (e.g. predictive gating must
beat plain temporal striding; surprise-weighted pooling must beat learned attention
pooling), or the neuroscience is unfalsifiable garnish. **Every brain-inspired block below
ships with its non-neuro twin as a mandatory baseline.**

**T2 — MACs ≠ on-device latency, and mobile SLAs are worst-case.** Half of the tempting
ideas cut FLOPs but *raise* wall-clock on a phone NPU because they introduce (i)
unstructured sparsity, (ii) data-dependent control flow, or (iii) memory-bound ops. Core
ML / TFLite reward **static, dense, structured** graphs. This is the deployment analog of
"high Brain-Score ≠ mechanistic model": a FLOP win that fights static-graph compilation is
worthless here. Two hard rules follow, and they are the most common way an "efficient"
paper dies in review:

- **Report measured on-device latency (Core ML / TFLite / ONNX-Runtime-Mobile), not MACs.**
  Every efficiency claim in this note is falsified by a MAC-only table.
- **Report worst-case, not average.** Adaptive-compute stories that only improve the mean
  do not improve the number a mobile app actually contracts on.

---

## 1. Where the compute and the losses actually are (source recap)

The four bottlenecks a good idea must attack, with current anchors:

| Tag | Bottleneck | Code | Share |
|-----|-----------|------|-------|
| **(a)** | Full conv backbone run **per frame** (×T=8) | `FastViT.forward` flattens `T·B` and runs the whole tower: `mobileviclip_tiny.py:1288-1294` (`x.reshape(T*B,3,H,W)` → `forward_embeddings`/`forward_tokens`/`conv_exp`) | **dominant share of video MACs** (analytic est., unprofiled ⚠) — the real target |
| **(b)** | Joint T·H·W global attention, T·H·W tokens (512 at T=8), export-hostile | `AttentionBlock` `:955` + `temporal_embedding` `:1026`; `MHSA` `:371`. ~~T=8 hard-coded at `:1017`~~ **no longer true — dynamic since PR #222**, now `t_frames = get_num_frames()` (`:1031`) | minority of MACs (analytic est. ⚠) but peak memory + the op that breaks Core ML lowering (`whitespace-directions.md §0`) |
| **(c)** | Lossy aggregator: unweighted mean over T frame embeddings | `FastViT.forward:1312-1318` (`x_new_stack.mean(0)`; the `self.temporal_pool` branch is the P3 attention-pool head, **now merged and switchable** via `temporal_pooling="attn"`) | discards temporal saliency — but see the round-2 sharpening below |
| **(d)** | Small-batch InfoNCE, frozen temperature | `VTC_VTM_Loss.vtc_loss` (`criterions.py`), global batch 512, temp fixed 1/100 | small-batch contrastive is quality-limited |

⚠ **The "92% / 8%" split is an unprofiled analytic estimate** (carried from the feasibility
review). `model-architecture-improvements.md:73` states only that per-frame stages 1–3
dominate total FLOPs — *"analytic, no measured profile yet"*. **Profile the per-frame vs
stage-4 MAC/latency split before Tier C's efficiency claim leans on it.**

> **Round-2 sharpening of (c) (2026-07-24, `neural-inspired-round2.md` F1–F3).** "Discards
> temporal saliency" understates it. F1 proves the head is **unconditionally order-blind**:
> mean pooling is permutation-invariant, so *no* reweighting of the frame set — attention pool,
> surprise pool, or any other — can recover frame order. F2 proves the additive
> `temporal_embedding` is **exactly annihilated** by the mean, so it contributes zero
> order-sensitivity; only *multiplicative* binding survives. The real gap is that the model has
> **no conjunctive what×when channel**, and the fix is a readout change (round 2 **R2**), not a
> better aggregator. Read every "(c)" claim below against this.

The temporal RepMixer — a parallel depthwise `Conv1d(k=3)` over time inside every
RepMixer block (`RepMixer.forward:596`, `reparam_conv_1d:533`, `mixer_1d:571`) — is
the one existing cross-frame op, and it is the natural host for several proposals below.
(Line numbers re-verified 2026-07-29 after PR #222; re-locate by symbol if they drift.)

---

## 2. Proposals, ranked

Ranking = (deployable win or novelty) × SOTA-defensibility ÷ (risk + engineering cost),
under T1/T2 and the repo's 8×V100 budget. Named **N1–N8** to avoid collision with the
P1–P10 backlog.

| # | Proposal | Neuro anchor | Attacks | Deployable win | Novelty | Verdict |
|---|----------|--------------|---------|----------------|---------|---------|
| **N1** | **Geometry-preserving distillation** (RSA/CKA + CCGP) folded into C1 | Kriegeskorte RSA; Kornblith CKA; Bernardi/Fusi CCGP | (c),(d) | none (train-time) | **High** | **Do — best free novelty** |
| **N2** | **Predictive-coding structured frame gating** (+ error-weighted pooling) | Rao & Ballard; PredNet | **(a)**; ~~(c)~~ ⚠ | **High** if structured | High (gating half) | **Gating half: headline candidate — high risk; GATE-0 ✅ closed 07-27 and found order ≈ worthless but frame *diversity* worth ≈13 R@1, which is the redundancy the gating half actually targets ⇒ premise survives, still unfunded. Pooling half (A1): demoted by round 2 §6.1 — order-blind, not a contribution** |
| N3 | **Magno/parvo two-stream** (slow high-res form + fast low-res motion) | SlowFast's literal basis; dorsal/ventral | (a) | High, reliable | Low | Safe deployment substrate under N1/N2 |
| N4 | **Graded temporal receptive windows** (dilation grows with depth) | Hasson temporal-window hierarchy | (c) | tiny, free | Low | Free ablation, not a headline |
| N5 | Adaptive-T via evidence accumulation (**coarse router only**) | Drift-diffusion; ACT/PonderNet | (a) avg | weak (worst-case) | Med | Router-of-static-graphs variant only |
| N6 | Divisive normalization (GDN layer) | Carandini & Heeger; Ballé GDN | — | minor | Low | **Quantization-robustness sub-study only** |
| N7 | Foveated / active token sampling | Mnih RAM; active vision | (a),(b) | ≈ token-merging | Low | Redundant with existing plan — fold in |
| N8 | Spiking / event-based (SNN) | Roy/Panda; Spikformer | — | **negative** | — | **Trap — rule out** |

**Recommended thesis:** one train-time quality contribution (**N1**) + one deployment
lever (**N2** as the ambitious headline, or **N3** as its low-risk fallback), with **N4**
as a free ablation. That yields a novel objective *and* a novel-but-deployable architecture
claim, each defensible independently — and both slot into the existing paper spine (§4).

---

### N1. Geometry-preserving distillation — the best free novelty (do this)

**Anchor.** RSA (Kriegeskorte et al., *Front. Syst. Neurosci.* 2008); CKA (Kornblith et
al., *ICML* 2019); CCGP / geometry of abstraction (Bernardi, Benna, Rigotti, Munuera, Fusi
& Salzman, *Cell* 2020). ML precedents: similarity-preserving KD (Tung & Mori, *ICCV*
2019); relational KD (Park et al., *CVPR* 2019); CRD (Tian et al., *ICLR* 2020);
correlation congruence (Peng et al., *ICCV* 2019); CLIP-KD (Yang et al., *CVPR* 2024).

**Idea.** C1 (`paper-plan.md`) already plans offline InternVideo2 distillation with
per-vector cosine + similarity-matrix KD. Cosine KD matches *where each embedding lands*;
it ignores the **relational structure** of the teacher's space — which is exactly what
retrieval, a ranking/geometry task, depends on. Add a loss that matches the teacher's
**inter-sample similarity geometry** (a CKA/RSA term over the batch's video–video,
text–text, and video–text Gram matrices), and — more ambitiously — its **CCGP abstraction
structure** (does the student preserve the teacher's linearly-separable, cross-condition-
generalizable factor geometry?).

**Why it is more than a KD footnote.** (i) A relational loss is an **O(B²) per-batch
signal** vs InfoNCE's O(B) — a denser constraint that *partially mitigates the small-batch
degradation (d)* the SigLIP move (P4/C3) also targets, from a different direction. (ii) The
CCGP angle — "preserve the teacher's abstraction manifold, not just its vectors" — is
genuinely under-explored in CLIP distillation and gives a clean falsifiable figure:
**student↔teacher CKA plotted against retrieval R@k across KD variants** (cosine-only, RKD-
distance, Tung–Mori SP-KD, ours). If structural KD helps R@k more than cosine KD *and* CKA
tracks R@k, that is a publishable result on its own.

- **Touchpoints:** the C1 loss in `criterions.py` (add a Gram-alignment term over the
  batch similarity matrices; reuse the teacher embeddings C1 already stores per clip).
  Zero new inference code.
- **Non-neuro twin (mandatory baseline, per T1):** RKD-distance and Tung–Mori SP-KD are
  the "same trick without the neuroscience framing" — N1 must beat them, or it reduces to
  a re-citation.
- **Risk:** clean CCGP needs labeled factor dichotomies that video-text data lacks; in
  practice CCGP may collapse to RSA/CKA matching. That is an acceptable fallback (RSA/CKA
  matching is still the novel-for-video-CLIP contribution). **Zero latency benefit** — this
  is purely a quality lever, fully orthogonal to deployment, which is why it pairs with N2/N3.
- **Eval:** full ZS retrieval suite vs cosine-KD-only at matched steps; the CKA-vs-R@k plot;
  ablate the video/text/cross Gram terms.

### N2. Predictive-coding structured frame gating (+ error-weighted pooling) — the headline bet

**Anchor.** Rao & Ballard (*Nat. Neurosci.* 1999, predictive coding in visual cortex);
Lotter, Kreiman & Cox (PredNet, *ICLR* 2017). ML cousins that a reviewer **will** cite, so
we must cite first: Deep Feature Flow (Zhu et al., *CVPR* 2017), Skip-Convolutions
(Habibian et al., *CVPR* 2021), DeltaCNN (Parger et al., *CVPR* 2022), and — closest of all
— **ResidualViT** (Soldelli et al., *ICCV* 2025), which does I-/P-frame interleaved
encoding (full encoder every (N+1)th frame + a lightweight residual tokenizer) for ~56%
cost cut at 1–3% accuracy drop.

**Idea (two coupled parts under one banner).**
1. *Structured gating (efficiency, attacks (a)).* Encode K=2–3 **keyframes** through the
   full FastViT tower; for intermediate frames, propagate the previous feature map and run
   only a cheap path — reinterpreting the temporal RepMixer `Conv1d` (`:589`,`:599`) as
   **prediction-error propagation** rather than plain temporal mixing. A tiny
   frame-level surprise signal decides, per frame, whether to skip whole backbone blocks.
2. *Error-weighted pooling (aggregator, attacks (c)).* Weight each frame embedding in the
   head (`:1312-1318`) by its prediction error — high-surprise frames carry more
   information than static ones. Built as **A1** (`neural-inspired-implementation.md §A1`,
   module `neural_inspired/temporal_surprise_pool.py`); 10k link-test passed 2026-07-24.

   > ⚠️ **Withdrawn framing (round 2 §6.1, 2026-07-24).** This bullet used to claim the
   > error-weighted pooling "**unifies the efficiency headline and the mean-pool fix into one
   > predictive-coding story**" and "gives (c) a brain-principled fix nobody proposed". **That
   > claim is retracted.** By F1 the head is order-blind under *any* set reweighting, so A1
   > reaches **appearance saliency only** — it cannot touch temporal order, which is what (c)
   > actually costs us. The two halves of N2 therefore do **not** unify: part 1 (gating) is an
   > efficiency bet, part 2 (pooling) is an unrelated, weak aggregator tweak that must beat the
   > plain attention pool to survive at all. Round 2's **R2 order-recoverable readout** is the
   > real fix for (c). A1 is explicitly **not** a paper contribution
   > (`neural-inspired-implementation.md §A1` novelty pin).

**The novelty is narrow and must be stated honestly (T1).** "Delta/predictive frame
encoding" is *done* (ResidualViT). What is open: ResidualViT is a **ViT** method; porting
delta-propagation into a **FastViT/MobileCLIP conv backbone** where the deltas are
depthwise-conv (cheaper and more quantization-friendly than ViT residual tokens), **under a
measured mobile INT8 latency budget**, is not done. The claim is not "delta encoding" but
"**structured, static, export-friendly predictive gating that reduces measured on-device
latency — not MACs — on a mobile conv backbone.**"

- **Critical design constraint (T2):** the win must come from **coarse structured gating**
  (skip whole blocks on flagged-static frames → static graph, dense ops) — **not** fine
  delta-sparsity, which NPUs do not accelerate and which can cut MACs while *raising*
  latency. Under real motion, naïve feature "prediction" needs alignment (flow/deformable
  warp) or the residual costs as much as recompute; the structured-gating form sidesteps
  this by choosing *which frames* to fully encode, not *which pixels*.
- **Touchpoints:** `FastViT.forward:1288-1301` (split frames by keyframe/intermediate role
  before the ×T flatten); stage-3/4 blocks; the RepMixer temporal branch as the
  cross-frame injector. Requires un-hard-coding T (P2/C2) first. Train from the existing
  checkpoint via self-distillation against the full-frame path — this is the same shape as
  **R4** in `paper-plan.md` (asymmetric frame encoding), now with an explicit
  predictive-coding rationale and the error-weighted-pooling extension.
- **Non-neuro twin (mandatory baseline):** uniform temporal striding to K frames + mean
  pool. N2 must beat *that trivial baseline* at equal on-device latency, and beat
  ResidualViT-style residual encoding, or it is not worth keeping.
- **Eval:** on-device latency (worst-case) vs R@k Pareto against uniform-K striding and
  ResidualViT; motion-heavy control (SSv2-MC) to detect prediction failure under motion.

### N3. Magno/parvo two-stream — the safe deployment substrate

**Anchor.** SlowFast (Feichtenhofer, Fan, Malik & He, *ICCV* 2019) explicitly cites the
retinal ~80% parvocellular / ~15–20% magnocellular split; deeper roots in two-stream
(Simonyan & Zisserman, *NeurIPS* 2014) and dorsal/ventral (Goodale & Milner, *TINS* 1992).

**Idea.** Run the **full** FastViT backbone on 2 high-res "slow/form" keyframes; run a
channel-slimmed, low-res "fast/motion" pathway over all 8 frames; fuse laterally (the
temporal RepMixer is the natural fusion site). You stop paying the expensive backbone ×8.
This lowers to **plain dense convs — export-clean** — and is the reliable, low-variance
latency cut.

- **Verdict:** low technical risk, but **low novelty** — a reviewer will say "mobile
  SlowFast" (and CLIP4Clip/X-CLIP are the CLIP-retrieval transfer baselines). Use it as the
  **deployment substrate under N1/N2**, not the headline. It is also the low-risk fallback
  if N2's predictive gating fails under motion: N2 and N3 attack (a) the same way (encode
  fewer full-resolution frames) — N2 chooses frames adaptively, N3 fixes the split.
- **Touchpoints:** `FastViT.forward` (two encoders / two resolutions), RepMixer fusion.

### N4. Graded temporal receptive windows — free garnish

**Anchor.** Hasson, Yang, Vallines, Heeger & Rubin (*J. Neurosci.* 2008, hierarchy of
temporal receptive windows); Lerner et al. (*J. Neurosci.* 2011).

**Idea.** Grow the temporal `Conv1d` dilation/kernel **stage-by-stage** (short windows
early, long late) instead of the fixed k=3 in every RepMixer (`:583-606`). One config
change, near-zero cost. Improves temporal features feeding the aggregator (c); does **not**
touch (a).

- **Verdict:** **incremental as a headline** — multiscale/dilated temporal convs predate
  the neuro story (TSM, Lin et al. *ICCV* 2019; MS-TCN, Farha & Gall *CVPR* 2019;
  Timeception, Hussein et al. *CVPR* 2019; MViTv2). Include it as a **free, well-motivated
  ablation** (most visible on SSv2-MC, which needs a long temporal receptive field);
  do not sell it as a contribution.

### N5. Adaptive-T via evidence accumulation — good idea, wrong metric

**Anchor.** Ratcliff drift-diffusion (*Psych. Rev.* 1978); Graves ACT (arXiv 2016);
PonderNet (Banino et al. 2021). Video precedents: AdaFrame (Wu et al., *CVPR* 2019), AR-Net
(Meng et al., *ECCV* 2020), SCSampler (Korbar et al., *ICCV* 2019), FrameExit (Ghodrati et
al., *CVPR* 2021).

**The crux (T2).** Learned per-clip halting improves T on *average* but: data-dependent
control flow breaks static-graph compilation, kills batching, and leaves **worst-case
latency — the mobile contract — unchanged**. It optimizes the wrong objective and fights
deployability at once. The only export-safe form is a **tiny router selecting among a few
pre-compiled static graphs (T=2/4/8)**, which discards the fine-grained ACT elegance but is
deployable. It also overlaps C2 (temporal-elastic training already makes T a knob) — the
router is the *inference-time policy* on top of C2's *train-time* elasticity.

- **Verdict:** include **only** as the coarse-router-of-static-graphs variant, **with
  worst-case latency reported**. Otherwise skip.

### N6. Divisive normalization — quantization-robustness angle only

**Anchor.** Carandini & Heeger (*Nat. Rev. Neurosci.* 2012, normalization as canonical
computation); DNN instantiation GDN (Ballé, Laparra & Simoncelli, *ICLR* 2016/2017).

**Idea.** Insert a GDN-style divisive-normalization layer (static, pointwise,
export-friendly). It attacks none of (a)–(d) directly and gives only a small accuracy
change. The **one non-obvious hook**: divisive normalization bounds/whitens activations,
which can improve **INT8 quantization robustness** — a genuine mobile concern tied to the
quantization line in `whitespace-directions.md §3`. That, not accuracy, is the only angle
worth a sub-study.

### N7. Foveated / active token sampling — redundant, fold in

**Anchor.** Recurrent attention / RAM (Mnih et al., *NeurIPS* 2014); active vision.

True glimpse policies are RL-trained, non-differentiable, **sequential** (serial glimpses
add latency), and export-hostile. The deployable residue is exactly **token
merging/pruning** — ToMe (Bolya et al., *ICLR* 2023), STTS (Wang et al., *ECCV* 2022),
TokenLearner (Ryoo et al., *NeurIPS* 2021) — which is already the P6/TempMe line in the
existing plan, and which targets the minority-cost stage (b) anyway. Foveating stage-4 tokens
caps the speedup at that small share. **The only version that touches the dominant per-frame
cost is foveating *which patches get
fully encoded per keyframe*** — which is a variant of N2, not a separate idea.

- **Verdict:** fold "saliency-weighted merging" into the existing token-merging line; do
  not open it as a brain-inspired contribution.

### N8. Spiking / event-based (SNN) — trap, rule out

**Anchor.** Roy, Jaiswal & Panda (*Nature* 2019); Spikformer (Zhou et al., *ICLR* 2023).

**Blunt verdict — negative expected value for this target.** (i) No commodity neuromorphic
silicon in phones (Loihi/Akida are not in an iPhone/Snapdragon SoC); on von-Neumann NPUs,
SNNs are **slower** (temporal unrolling over spike timesteps) and less accurate than dense
INT8. (ii) The input is dense RGB video, not an event-camera stream, so the sparsity
premise does not hold. (iii) Surrogate-gradient training lags CLIP-scale accuracy. The
repo's hygiene list (`paper-plan.md §6`) already flags removing the leftover `spikingjelly`
import — that instinct is correct. **Story-only; rule out.**

---

## 3. How these slot into the existing paper spine

The existing spine (`paper-plan.md`) is C1 (Video Dataset Reinforcement) + C2
(temporal-elastic distillation) + C3 (sigmoid loss, attention-pool head). The brain-inspired
mechanisms attach as follows — **no new parallel project**:

| This note | Attaches to | Nature of the addition |
|-----------|-------------|------------------------|
| **N1** geometry-preserving KD | **C1 loss** | New distillation term (RSA/CKA + CCGP) beside the planned cosine + sim-matrix KD; adds the CKA-vs-R@k figure and the "preserve the abstraction manifold" story |
| **N2** predictive gating + error-weighted pooling | **R4** (asymmetric encoding) + **C3** (pooling) | Gives R4 a predictive-coding rationale. ⚠ **Round 2 §6.1 retracts "the surprise signal unifies both"** — the pooling half (A1) is order-blind and attaches to C3 as a bare ablation row only; the halves are independent bets |
| **N3** two-stream | R4 fallback | Low-risk deployable substrate if N2 fails under motion |
| **N4** graded temporal windows | C3 / backbone | Free ablation on the temporal RepMixer |
| **N5** adaptive-T router | C2 | Inference-time policy over C2's train-time elasticity; report worst-case |
| **N6** divisive norm | `whitespace §3` quantization | Quantization-robustness ablation only |

**Positioning table (non-negotiable, from the feasibility review):** any efficiency claim
must be benchmarked against **TempMe** (Shen et al., *ICLR* 2025), **ResidualViT** (*ICCV*
2025), **EVS** (Efficient Video Sampling, 2025), **InternVideo2** (teacher/ceiling), and
base **MobileViCLIP** — reporting **mobile latency + R@k** on MSR-VTT / DiDeMo /
ActivityNet and the classification-MC sets.

## 4. Suggested sequencing (folds into the paper-plan weeks)

```
Alongside C1 (teacher pass):  N1 geometry-KD term — train-time only, cheapest novelty,
                              cannot regress inference. Land the CKA-vs-R@k figure early.
Alongside C3 A/Bs:            N4 graded temporal windows (free ablation).
After GATE-0, then R2/R3:     N2 as the R4 self-distillation fine-tune — structured gating
                              + error-weighted pooling; gate on worst-case on-device latency.
                              N3 stands ready as the low-risk substitute if N2 breaks on SSv2-MC.
Opportunistic:               N5 coarse router (worst-case reported); N6 quantization ablation.
Ruled out:                   N7 (fold into token-merging), N8 (SNN).
```

**Kill criteria (per T1/T2):**
- N1 must beat RKD-distance and Tung–Mori SP-KD on aggregated R-Sum, or it is a re-citation.
- N2 must beat uniform-K striding **and** ResidualViT-style encoding at **equal measured
  on-device latency**, or it collapses back to future work (same verdict rule as R4).
- N5 must improve **worst-case** latency, not just average, or drop it.

---

## 5. References (SOTA + neuroscience basis)

**Base / must-cite SOTA:**
- MobileViCLIP — Yang et al., *ICCV* 2025 (arXiv:2508.07312) — this repo's base.
- TempMe — Shen et al., *ICLR* 2025 (arXiv:2409.01156) — efficient text-video retrieval SOTA.
- ResidualViT — Soldelli et al., *ICCV* 2025 (arXiv:2509.13255) — I/P-frame encoding; N2's closest prior art.
- EVS (Efficient Video Sampling), 2025 (arXiv:2510.14624) — training-free static-patch pruning baseline.
- InternVideo2 — Wang et al., 2024 (arXiv:2403.15377) — teacher/ceiling.

**Neuroscience anchors:**
- Predictive coding — Rao & Ballard, *Nat. Neurosci.* 1999; PredNet — Lotter, Kreiman & Cox, *ICLR* 2017.
- Magno/parvo two-stream — SlowFast, Feichtenhofer et al., *ICCV* 2019; two-stream, Simonyan & Zisserman, *NeurIPS* 2014; dorsal/ventral, Goodale & Milner, *TINS* 1992.
- Temporal receptive windows — Hasson et al., *J. Neurosci.* 2008; Lerner et al., *J. Neurosci.* 2011.
- Representational geometry — RSA, Kriegeskorte et al., *Front. Syst. Neurosci.* 2008; CKA, Kornblith et al., *ICML* 2019; CCGP, Bernardi et al., *Cell* 2020.
- Evidence accumulation — Ratcliff, *Psych. Rev.* 1978; ACT, Graves, arXiv 2016; PonderNet, Banino et al., 2021.
- Divisive normalization — Carandini & Heeger, *Nat. Rev. Neurosci.* 2012; GDN, Ballé et al., *ICLR* 2016/2017.
- Active vision — RAM, Mnih et al., *NeurIPS* 2014.
- SNNs — Roy, Jaiswal & Panda, *Nature* 2019; Spikformer, Zhou et al., *ICLR* 2023.

**ML distillation / efficient-video precedents (N1/N2 baselines):**
- RKD — Park et al., *CVPR* 2019; Similarity-Preserving KD — Tung & Mori, *ICCV* 2019; CRD — Tian et al., *ICLR* 2020; Correlation Congruence — Peng et al., *ICCV* 2019; CLIP-KD — Yang et al., *CVPR* 2024.
- Deep Feature Flow — Zhu et al., *CVPR* 2017; Skip-Convolutions — Habibian et al., *CVPR* 2021; DeltaCNN — Parger et al., *CVPR* 2022; TeSTra — Zhao & Krähenbühl, *ECCV* 2022; MeMViT — Wu et al., *CVPR* 2022.
- Multiscale temporal — TSM, Lin et al., *ICCV* 2019; MS-TCN, Farha & Gall, *CVPR* 2019; Timeception, Hussein et al., *CVPR* 2019; MViTv2, Li et al., *CVPR* 2022.
- Adaptive frames / tokens — AdaFrame, Wu et al., *CVPR* 2019; AR-Net, Meng et al., *ECCV* 2020; SCSampler, Korbar et al., *ICCV* 2019; FrameExit, Ghodrati et al., *CVPR* 2021; OCSampler, Lin et al., *CVPR* 2022; ToMe, Bolya et al., *ICLR* 2023; STTS, Wang et al., *ECCV* 2022; TokenLearner, Ryoo et al., *NeurIPS* 2021.

*Citations are correct anchors (venue/year); verify page numbers at submission. Line/file
references checked 2026-07-23 against `models/mobileviclip_tiny.py`.*
