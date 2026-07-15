# Model-Architecture Improvement Proposals (code-grounded)

**Scope.** This note answers: *given the model structure as implemented in this repo,
what should we change in the model (or its training objective) to make it more
efficient and/or more accurate?* It complements the existing notes:

- `notes/whitespace-directions.md` — data pipeline, temporal hard negatives,
  quantization, frame selection (mostly *around* the model);
- `notes/data-quality-caprl-scheme.md` — caption/data quality (data side).

This note is the **model side**: architecture, token mixing, pooling, losses.
Every claim below is anchored to a file/line in this repo (checked 2026-07-15;
re-locate by symbol if files drift).

---

## 1. What the model actually is (source-level recap)

Canonical model = `MobileViCLIP_Small` (`models/mobileviclip_small.py`), the
Tiny variant is the same code at smaller width/depth (`models/mobileviclip_tiny.py`).

**Vision tower** — MobileCLIP's FastViT/MCi image backbone run **per frame**, with
three video-specific insertions:

1. **Temporal RepMixer branch**: every `RepMixer` gets a parallel depthwise
   `Conv1d(k=3)` over the time axis (`mixer_1d`/`norm_1d`/`layer_scale_1d`),
   reparameterized into a single `Conv1d` for inference
   (`mobileviclip_tiny.py:458-656`). This is the paper's "temporal structural
   reparameterization".
2. **Joint T·H·W global attention** in stage 4: `MHSA` flattens all
   `T×H×W = 8×8×8 = 512` tokens and runs full self-attention
   (`mobileviclip_tiny.py:336-399`), plus a learnable
   `temporal_embedding` of shape `(1, C, 8)` (`AttentionBlock`, `:989`).
3. **Head**: `conv_exp` (MobileOne+SE) → `GlobalPool2D` per-frame mean-pool +
   linear proj → **plain mean over the 8 frame embeddings**
   (`FastViT.forward`, `mobileviclip_tiny.py:1253-1263`).

**Text tower** — MobileCLIP text transformer, frozen except the projection
(`freeze_text=True`, `open_text_projection=True` in
`scripts/pretraining/clip/mobileviclip_small/config.py`). For Small
(`mobileclip_s2.json`) this is the 12-layer, 512-dim "base" text encoder —
**~63M params, i.e. bigger than the ~36M vision tower**.

**Objective** — VTC (InfoNCE) only (`VTC_VTM_Loss.vtc_loss`,
`models/criterions.py:93`), **fixed** temperature 1/100
(`temp == temp_min == 0.01`, so the clamp makes it constant), global batch 512
(8×64), 3 epochs on InternVid-10M-FLT, LiT-style (vision open, text locked).

**Latent constraints baked into the code:**

- `T = 8` is **hard-coded** in every temporal reshape
  (`x.reshape(8, int(x.shape[0]/8), ...)` in `MHSA.forward`,
  `RepMixer.forward`, `AttentionBlock.forward`) and in
  `temporal_embedding = nn.Parameter(torch.zeros(1, C, 8))`.
- Attention is **naive** (explicit `softmax(q@kᵀ)@v`); the
  `use_flash_attn / use_fused_rmsnorm / use_fused_mlp` flags are plumbed
  through `basic_blocks(...)` and then **ignored**.
- Dead-but-present code that several proposals below can reuse:
  a complete SigLIP-style loss `VTC_VTM_Loss.vtc_loss_sigmoid`
  (`criterions.py:133`, with learnable `t`/`b` hooks — never called),
  `get_sim_sigmoid` (`criterions.py:62`), `AttentionPoolingBlock` /
  `AttentiveBlock` / `CrossAttention` (`mobileviclip_tiny.py:1396-1487`),
  `open_text_lora` config flag.
- A real **freeze-logic bug**: `hasattr(name, '_1d')` on a *string* is always
  `False` (`mobileviclip_tiny.py:55`, same in small), so
  `open_vision_clip_projector` never unfreezes the projector via that branch —
  only the `"mixer_1d" in name or ...` branch works. Any
  freeze-based experiment (e.g. temporal-only fine-tuning) should fix this first.

**Where the compute actually goes** (analytic, no measured profile yet):
per-frame conv stages 1–3 dominate **total** video FLOPs (cost ∝ T = 8 full
backbone passes), while the stage-4 joint attention (512 tokens, 2–4 blocks)
dominates **peak activation memory** and is the most **export-hostile** op
(cf. `notes/whitespace-directions.md §0`: the only Core ML export script skips
every temporal op). Efficiency work should therefore attack (a) the ×T
per-frame cost for throughput and (b) the joint attention for deployability —
they are different problems.

---

## 2. Proposals, ranked

Ranking = expected gain × SOTA evidence ÷ (risk + engineering cost), given the
repo's state (replication not yet finished; 8×V100 budget).

| # | Proposal | Type | Inference cost | Expected effect | Risk |
|---|----------|------|----------------|-----------------|------|
| P1 | SDPA/Flash attention + fused kernels | eng. fix | ↓ (GPU) | faster train/eval, identical math | ~0 |
| P2 | Un-hard-code T; frame-count-flexible training | eng.+train | = | variable-T eval, long-video, prereq for P5/P6 | low |
| P3 | Attention-pool the frame axis (replace mean-over-T) | arch (head) | +ε | better temporal aggregation, biggest acc/FLOP ratio | low |
| P4 | SigLIP sigmoid loss + learnable temp/bias | objective | = | better acc at batch 512; robust to noisy captions | low |
| P5 | Video dataset-reinforcement: distill InternVideo2 (+MobileCLIP2 recipe) | objective | = | **largest expected accuracy jump** | med (storage/teacher infra) |
| P6 | Temporal token merging in stage 4 (TempMe/ToMe-style) | arch | ↓ | −30–50% stage-4 cost, ≈acc | med |
| P7 | Asymmetric frame encoding (keyframe full path, deltas cheap path) | arch | ↓↓ | attack the ×T cost directly | high (research bet) |
| P8 | Text-tower rebalance (mct-style small text + KD) | arch (text) | ↓ size | −40–50M params model size | med |
| P9 | Divided space-time attention (export-friendly stage 4) | arch | = / ↓ | unblocks on-device export | med |
| P10 | Matryoshka embedding head | objective | = | cheap on-device ANN at 64–256 dims | low |

The sections below give the reasoning, code touchpoints, and eval for each.

---

### P1. Use SDPA/Flash attention (pure engineering, do first)

`MHSA.forward`, `CrossAttention.forward`, and `Attention_global_residual.forward`
all materialize the full `N×N` attention matrix. Replace the body with
`F.scaled_dot_product_attention`. The flags for this (`use_flash_attn`) already
exist and are ignored — wire them up or just make SDPA unconditional.

**Validation correction (2026-07-15):** the pinned env is **PyTorch 1.13.1**
(`mobileviclip.yml`), which predates SDPA (added in 2.0). So this requires
either a torch ≥2.0 env bump (cu117/cu118 wheels exist; verify DeepSpeed/apex
compat) or `xformers.ops.memory_efficient_attention` on 1.13. Also: the Flash
kernel needs sm80+; **V100 (sm70) gets the memory-efficient kernel**, which
still removes the N×N materialization — the win is real but it is a
memory+moderate-speed win, not a Flash-class speedup.

- **Touchpoints:** `models/mobileviclip_{small,tiny}.py` (`MHSA`,
  `CrossAttention`); keep the reshape choreography, swap only the attention core.
- **Payoff:** faster + leaner pretraining and GPU eval at N=512 tokens;
  numerically equivalent (softmax in fp32 inside SDPA is *more* stable than the
  current fp16 softmax under V100 `use_bf16=False`).
- **Eval:** bitwise-close logits on a fixed batch; step-time before/after.

### P2. Remove the hard-coded T=8; train frame-count-flexible

Every temporal op assumes exactly 8 frames, so the model cannot: encode images
(T=1) with the video tower, eval with 16 frames on long-video benchmarks, or
amortize cost with fewer frames on easy clips. Fix by threading `T` (from
`encode_vision`, which knows `image.shape[1]`) into the blocks, and
interpolating `temporal_embedding` (`F.interpolate` over the last axis) the same
way `interpolate_pos_embed` already handles the checkpoint path. The 1D RepMixer
is a depthwise conv — already T-agnostic; only the reshapes and the embedding pin T.

Then (cheap, high-leverage): **randomize T ∈ {4, 8, 12, 16} during training**
(FlexiViT-style, applied to time). SOTA long-video work (LiteFrame 2026,
frame-scaling for video LLMs) shows frame-count flexibility at train time is
what makes test-time frame scaling work.

- **Touchpoints:** `MHSA.forward`, `RepMixer.forward`,
  `AttentionBlock.forward`, `temporal_embedding` init; dataset already
  parameterizes `num_frames`.
- **Payoff:** frames-vs-accuracy curve becomes a *deployment knob* (4 frames on
  device, 16 for offline indexing); prerequisite for P6/P7 and for the
  frame-selection direction in `whitespace-directions.md §4`.
- **Eval:** MSR-VTT/DiDeMo R@1 at T=4/8/16 from one checkpoint vs the fixed-8 baseline.

### P3. Replace mean-over-frames with a tiny temporal attention pool

The final video embedding is an **unweighted mean of 8 frame embeddings**
(`x_new_stack.mean(0)`). This is the cheapest possible aggregator and provably
discards temporal saliency: a 2-second relevant action in an 8-frame window is
averaged 1:7 with background. The file *already defines* an
`AttentionPoolingBlock` (mean-query cross-attention, `mobileviclip_tiny.py:1480`)
that the canonical MobileViCLIP model never instantiates — the InternVideo2
backbones in this repo use their own copies of the same class as their
`clip_projector` head (`internvideo2.py:484`), which is direct in-repo evidence
the pattern works. Perception Encoder likewise uses an attention-pool head.

Concretely: keep per-frame `GlobalPool2D`, then run **one** cross-attention
block over the T frame embeddings (query = mean, keys/values = frames), i.e.
attention over **8 tokens of dim 512** — on the order of 1e-5 of total video
FLOPs, ~1M params, trivially exportable (it's a tiny matmul).
Optionally also try pooling over T×H×W stage-4 tokens (more expressive, still
cheap: 512 tokens once per video, not per frame).

- **Touchpoints:** `FastViT.forward` + `GlobalPool2D`; reuse
  `AttentionPoolingBlock`. Init the output proj to zero (the class already
  zero-inits `Attention_global_residual.proj`) so training starts exactly at
  the mean-pool baseline — strictly-no-regression trick.
- **Payoff:** the standard finding (attention pool vs mean pool in CLIP-family
  heads) is +0.5–1.5 R@1 for near-zero cost; also the natural place where
  temporal-hard-negative fine-tuning (`whitespace-directions.md §2`) gets its
  capacity.
- **Eval:** MSR-VTT/DiDeMo/ANet R@1 and K400 ZS vs mean-pool at identical
  budget; ablate frame-level vs token-level pooling.

### P4. SigLIP-style sigmoid loss + learnable temperature/bias

Two objective-level facts about the current setup work against it:

1. **Global batch is 512.** InfoNCE quality is strongly batch-size dependent;
   SigLIP (Zhai et al. 2023) was designed exactly for this regime and
   *outperforms softmax at small batch*. The repo already contains a **complete,
   never-called implementation** — `VTC_VTM_Loss.vtc_loss_sigmoid`
   (`criterions.py:133`) with learnable logit-scale `t` and bias `b` hooks —
   so this is a wiring change in `MobileViCLIP_Small.forward`, not new code.
   One caveat: it builds targets from `torch.eye` and ignores `idx`, so the
   duplicated-caption positives that `vtc_loss` handles via `get_mask`
   (`criterions.py:270`) must be ported (mask off-diagonal positives instead
   of treating them as negatives).
2. **Temperature is frozen at 1/100** (`temp == temp_min`), inherited from
   ViCLIP. CLIP/SigLIP learn temperature (and SigLIP a bias); MobileCLIP2's
   ablations show temperature tuning of contrastive+distillation losses is
   worth up to ~1–2% on its own.

InternVid captions are noisy and near-duplicated (multiple clips of one video
share captions — the `idx`-based `get_mask` in `criterions.py` already
struggles with this); sigmoid's per-pair independence handles multi-positives
more gracefully than batch-softmax.

- **Touchpoints:** `mobileviclip_small.py` (`forward` → call
  `vtc_loss_sigmoid`; add learnable `self.logit_scale`/`self.logit_bias` with
  SigLIP init t=log 10, b=−10; free the `temp` clamp), `criterions.py`
  (port `get_mask` into the sigmoid targets), config `criterion.loss_weight`.
- **Payoff:** zero inference cost; expected +0.5–2 R@1 at this batch size.
- **Eval:** A/B at matched steps on the standard harness. Cheap to kill if flat.

### P5. Video dataset-reinforcement: distill from InternVideo2 (the MobileCLIP recipe, applied to video)

The single most-proven way to make a *small* CLIP-family model strong is not an
architecture tweak — it is **multi-modal reinforced training**: MobileCLIP
(CVPR'24) and MobileCLIP2 (TMLR'25) store teacher-ensemble embeddings +
synthetic captions *in the dataset* (DataCompDR/DFNDR) and train the student
with contrastive + distillation losses; InternVideo2-S/B themselves are
distilled from InternVideo2-1B. The vision backbone here **is** MobileCLIP's —
but the video training pipeline (`tasks/pretrain.py`, VTC-only) uses none of
the recipe that made that backbone good in the first place.

Concretely, a **video DR pass over InternVid-10M-FLT**:

1. Offline, once: run InternVideo2-L (or 1B if affordable) over the training
   clips; store video embeddings + text embeddings for both real and
   (CapRL-recaptioned, see `notes/data-quality-caprl-scheme.md`) synthetic
   captions. Storage at 512-d fp16 ≈ 1 KB/clip → ~10–20 GB for 10M clips —
   entirely feasible.
2. Train with `L = L_VTC + λ₁·(1 − cos(z_student, z_teacher)) +
   λ₂·KL(sim_teacher ‖ sim_student)` (similarity-matrix KD over the batch —
   the MobileCLIP formulation; both terms are ~10 lines in `criterions.py`).
3. Multiple captions per clip (real + synthetic) sampled per step — the
   dataset already has `caption_augmentation` hooks (`dataset/pt_dataset.py`).

- **Payoff:** on the image side this recipe is worth **+2–5 points** at equal
  student size and *reduces* needed steps (teacher does the hard mining).
  Nothing published does this for on-device *video* CLIP yet — it is also the
  clearest paper-shaped contribution available to this repo.
- **Risk/cost:** teacher inference over the corpus (~1 GPU-week class for
  InternVideo2-L over 10M×8-frame clips — measure on a slice first);
  embedding storage versioning. No inference-time cost whatsoever.
- **Eval:** the full ZS suite vs VTC-only at matched steps; ablate λ₁/λ₂/caption mix.

### P6. Temporal token merging in the attention stage

Stage 4 attends over 512 tokens of which the temporal axis is highly redundant
(adjacent frames of a 1fps-sampled clip are near-duplicates at 8×8 resolution).
TempMe (ICLR'25) shows merging temporal tokens progressively in CLIP-based
video retrieval cuts GFLOPs ~50% *and improves* R-Sum; ToMe/vid-TLDR show
training-free variants. Here: insert a merge step (bipartite soft matching on
key similarity, merging only across the T axis first) before each
`AttentionBlock`, unmerge after stage 4 (or never — the head pools anyway, so
"merge-and-pool" is natural with P3's token-level pooling).

- **Touchpoints:** `AttentionBlock.forward` / `basic_blocks` (stage-4 path
  only); needs P2 (dynamic token counts kill the hard-coded reshapes).
- **Payoff:** bounded — stage 4 is a minority of *total* FLOPs (the ×8
  per-frame conv stages dominate), but it cuts the peak-memory / most
  export-hostile op, and composes with P9. Honest framing: latency win on
  device, modest GPU win.
- **Eval:** R@1 vs %-tokens-merged curve; measure, don't assume, the on-device
  latency (same kill-criterion discipline as quantization in
  `whitespace-directions.md §3`).

### P7. Asymmetric frame encoding (research bet)

The real ×T cost is running the **full conv backbone on every frame**. The
video-efficiency literature attacks exactly this: DeltaCNN/Skip-Conv (sparse
frame differences), SparseTem (temporal continuity in CNN encoders),
SlowFast-style two-rate pathways, and 2026 work (LiteFrame; MeToM CVPR'26)
internalizing frame compression in the backbone. Proposal shaped to this
architecture: encode **K=2–4 keyframes** through all 4 stages; run the other
frames only through stages 1–2 (cheap, high-res motion evidence) and inject
them into the keyframe stream at stage 3 via the *existing* temporal RepMixer
(which is exactly a cross-frame mixer). Roughly halves video encode FLOPs at
K=4.

- **Touchpoints:** `FastViT.forward` (split batch by frame role), stage-3/4
  blocks; needs P2 first. Train from the existing checkpoint with distillation
  (P5 loss) against the full-frame teacher — self-distillation makes this a
  fine-tune, not a from-scratch run.
- **Risk:** genuine research risk (motion-heavy benchmarks like SSv2-MC may
  drop); highest novelty of the efficiency items — "temporal
  reparameterization + asymmetric frame paths" is a coherent follow-up story
  to the MobileViCLIP paper itself.
- **Eval:** FLOPs/latency vs R@1 Pareto against frame-subsampling baseline
  (T=4 uniform) — it must beat *that* trivial baseline to be worth keeping.

### P8. Text-tower rebalance (the elephant in the checkpoint)

MobileCLIP-S2's text encoder is **~63M params vs ~36M for the vision tower** —
the majority of on-device *model size*, even though per-query text encoding is
cheap. MobileCLIP-S0 ships a 4-layer conv-augmented text tower ("mct",
`configs/mobileclip_s0.json`, implemented in
`models/backbones/internvideo2/mobileclip/modules/text/repmixer.py`) that loses
only ~1–2 points on image benchmarks when trained with dataset reinforcement.
Options, cheapest first: (a) swap Small's text config to the mct tower and
distill from the frozen 12-layer teacher's text embeddings (pure KD — the text
side is frozen anyway, so this doesn't perturb the video recipe); (b)
TinyCLIP-style weight inheritance (drop alternating layers, then KD).

- **Payoff:** −40–50M params (≈ −45% total size) for a likely ≤1 R@1 cost —
  for on-device deployment this is the best size lever available, bigger than
  INT8 quantization of the vision tower.
- **Eval:** text-side KD loss convergence; full retrieval suite; model size /
  load time on device.

### P9. Export-friendly stage 4: divided space–time attention

Direct response to the deployment finding in `whitespace-directions.md §0`
(the only export script silently drops all temporal ops). Joint T·H·W attention
is the op that breaks Core ML/TFLite lowering. TimeSformer's ablation shows
divided space-time attention is **both more accurate and cheaper** than joint
(best K400/SSv2 accuracy; joint OOMs at 32 frames — which also caps P2's
frame scaling). This suggests replacing it with **per-frame spatial
attention (N=64) + the already-present 1D temporal RepMixer** (possibly one
extra temporal conv per attention block). Per-frame 64-token attention is
static-shape, batch-friendly, and exports today.

- **Touchpoints:** `MHSA.forward` (drop the T-flatten; attend per frame),
  `AttentionBlock` (keep `temporal_embedding` on the conv path).
- **Payoff:** unblocks Step-0 export; also cuts attention FLOPs ~8× (64² vs
  512² per frame-group). Cost: some temporal modeling capacity — measure it;
  if ZS temporal benchmarks drop, a hybrid (joint attention in the last block
  only) is the fallback.
- **Eval:** SSv2-MC / TVBench (temporal-sensitive) + standard retrieval; a
  successful Core ML export with all temporal ops present is itself the deliverable.

### P10. Matryoshka embedding head (free deployment win)

Add MRL loss over nested prefixes (64/128/256/512-d) of both towers'
embeddings — a loop over `vtc_loss` on sliced, re-normalized projections.
On-device retrieval then runs ANN at 64–128 dims (4–8× smaller index, faster
search) and re-ranks at 512. No architecture change, no inference cost,
pairs with P4/P5.

---

## 3. Suggested sequencing

```
Phase 0 (with replication):   P1 (SDPA) + freeze-bug fix           [days]
Phase 1 (first improvements): P4 (sigmoid+temp) → P3 (attn pool)   [1-2 wks, one A/B each]
Phase 2 (the big swing):      P5 (video DR distillation), with P10 folded in
Phase 3 (efficiency track):   P2 (flex-T) → P9 (export) → P6 (merging) → P7 (bet)
Phase 8 (size):               P8 whenever a deployment target demands it
```

P3+P4+P5 are mutually compatible and share the same eval harness; the
efficiency track (P2→P9→P6→P7) is independent and gated by the on-device
export prerequisite already documented in `whitespace-directions.md §0`.

---

## 4. References (SOTA basis)

- MobileViCLIP, ICCV 2025 — arXiv:2508.07312 (this repo's base).
- MobileCLIP, CVPR 2024 (arXiv:2311.17049) / **MobileCLIP2**, TMLR 2025
  (arXiv:2508.20691) — multi-modal reinforced training / DR datasets.
- SigLIP, ICCV 2023 (arXiv:2303.15343); SigLIP 2 (arXiv:2502.14786) — sigmoid
  loss, small-batch behavior.
- **TempMe**, ICLR 2025 (arXiv:2409.01156) — temporal token merging for
  text-video retrieval (−95% tokens, −51% GFLOPs, +R-Sum); ToMe (ICLR 2023),
  vid-TLDR (CVPR 2024).
- InternVideo2 (arXiv:2403.15377) — teacher; S/B variants distilled from 1B;
  attention-pooling head.
- TimeSformer (ICML 2021) — divided vs joint space-time attention ablation.
- FlexiViT (CVPR 2023) — train-time size flexibility (applied here to T).
- DeltaCNN (CVPR 2022), Skip-Convolutions (ICCV 2021), SparseTem
  (arXiv:2410.20790), LiteFrame (arXiv:2605.17260), MeToM (CVPR 2026) —
  frame-redundancy computation skipping.
- TinyCLIP (ICCV 2023) — weight-inheritance text-tower compression.
- Matryoshka Representation Learning (NeurIPS 2022).
- Perception Encoder (arXiv:2504.13181) — attention-pool head at the frontier.
