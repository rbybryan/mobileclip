# Whitespace Directions — On-Device Video-CLIP (Phase 3 candidates)

Concrete, code-grounded plans for the two research directions that a July-2026
literature sweep found to be **essentially unpublished** for contrastive
video-text (video-CLIP) dual encoders, and that are directly reachable from this
repo's MobileViCLIP baseline. This turns the "Improve" bullets of
[`replication-plan.md`](replication-plan.md) Phase 3 into actionable plans.

Survey basis: `docs/clip-2025-progress-report.md` (§7 "video-CLIP 新进展" + 2026
snapshot). Two whitespaces were confirmed by **six independent negative
searches**; a third (token reduction for retrieval encoders) is thin.

| Direction | Status as of 2026-07 | Reachable from MobileViCLIP? |
|---|---|---|
| **A. Quantization of a video-text dual encoder** | **Zero papers.** All "video + quantization" hits are generative/diffusion/tokenizer/recsys; nearest is LQA (*image*-text VLM edge quant). | Yes — reparameterized graph is quant-friendly |
| **B. Streaming / real-time contrastive video retrieval** | **None.** All 2026 "streaming video" work is generative video-LLM/agents (StreamSense, StreamBridge). | Yes — needs a causal temporal branch |
| Token reduction for retrieval encoders | Thin (efficiency work is on pretraining/understanding: ClusterSTM, AdaFocus) | Yes (secondary) |
| ~~Temporal grounding via contrastive alignment~~ | **Crowded** — CVA (CVPR'26), GranAlign (AAAI'26), DART (ECCV'26), HieraMamba (CVPR'26), MASRA, CoSTL… | Avoid — novelty must be very sharp |

> Recommendation: do **A (quantization)** first — smaller, self-contained, an
> immediate on-device win — then **B (streaming)**, a larger architectural change.

---

## 0. Shared prerequisites (do once)

1. **FP16 baseline + eval harness.** Reproduce MobileViCLIP-Small/Tiny zero-shot:
   MSR-VTT / DiDeMo / ActivityNet T2V+V2T R@1, and K400 / UCF101 / HMDB51 action
   recognition. All eval corpora are already wired in `configs/data.py`
   (`msrvtt_1k_test`, `didemo_ret_test`, `anet_ret_val`, `k400_act_val`, …).
2. **On-device latency/energy harness.** Core ML export + measurement on the
   target phone/tablet, reproducing the paper's iPad-Air numbers
   (Small ≈ 37.8 ms). Report ms **and** energy — both are the deliverable.
3. **Frozen FP16 teacher.** Keep the reproduced FP model as a teacher for
   metric-preservation / KD in both plans.

---

## 1. Plan A — Retrieval-aware quantization of MobileViCLIP

**Goal.** INT8 (then INT4-weight) MobileViCLIP running on device with **≤ 1 pt**
zero-shot R@1 drop and a measurable latency / energy / model-size win.

**Why it's open.** No PTQ/QAT for a video-text *contrastive dual encoder* exists
(2026-07). CLIP-family models are quantization-sensitive because retrieval
depends on the **direction of the L2-normalized embedding (cosine)**, not raw
logits — so naïve per-layer-MSE PTQ preserves activations but still **misranks**.

**Why MobileViCLIP is unusually quant-friendly.** After
`RepMixer.reparameterize()` (`models/mobileviclip_small.py` L624–695) the
train-time multi-branch mixer collapses to a **single depthwise `reparam_conv`
(2D) + `reparam_conv_1d` (1D)**; MobileOne / RepLKNet blocks likewise fold BN
into one conv. The deployed graph is therefore plain conv/linear — no train-time
branches, no separate BN-folding at quant time — which is close to ideal for PTQ.

### Approach

1. **Precision ladder.** `W8A8` (per-output-channel weights, percentile/per-tensor
   activations) → `W4A8` → `W4A16`. Start with PTQ; escalate to QAT only where a
   tier fails the R@1 budget.
2. **Sensitivity map first.** Quantize one layer at a time and measure the
   **cosine drift of the final embedding** (not layer MSE). Expected hot spots:
   the depthwise convs (spatial **and** the temporal `reparam_conv_1d`), the
   attention stage (`AttentionBlock`, L1030+, softmax + `temporal_embedding`),
   and the projection heads (`GlobalPool2D.proj` L1301 + the text projection).
   Keep the top-k most sensitive layers in higher precision (**mixed precision**).
3. **Depthwise-conv handling.** Per-output-channel weight scales are mandatory;
   migrate activation outliers with a SmoothQuant-style scale folded into the
   preceding conv. The **temporal `reparam_conv_1d`** is the novel, untested piece
   — profile it explicitly (it carries all temporal information).
4. **Retrieval-aware calibration (the novelty).** Calibrate against the *ranking*,
   not just per-layer error:
   `L = λ1·Σ layer-MSE + λ2·(1 − cos(z_fp, z_q)) + λ3·KL(sim_fp ‖ sim_q)`,
   where `z` is the normalized embedding and `sim` is the image-text similarity
   matrix on a held-out InternVid batch. This preserves what R@1 depends on.
5. **QAT recovery.** If a W4 tier drops > 1–2 pts, run short QAT from the FP16
   teacher using the repo's `VTC_VTM_Loss.vtc_loss` (`models/mobileviclip_small.py`
   L89, L124) **plus** an embedding/similarity distillation term (LSQ scales,
   straight-through estimator). This reuses the "reinforced training / teacher"
   philosophy the MobileCLIP lineage already relies on — here the FP model is the
   teacher.
6. **Calibration data.** A few thousand `internvid_all` clips at 8 frames
   (`configs/data.py`) for real temporal activation statistics.
7. **Deploy & measure.** Core ML / TFLite INT8; report R@1 vs FP16 at each tier +
   on-device latency / energy / model size.

**Novelty.** First metric-preserving PTQ/QAT for an on-device video-text dual
encoder; exploits the reparameterized deploy graph; retrieval-aware calibration +
FP-teacher similarity distillation; first study of quantizing a temporal 1D-conv
mixing branch.

**Eval.** ΔR@1 (MSR-VTT / DiDeMo / ANet ZS) and K400 ZS at W8A8 / W4A8 / W4A16 vs
FP16; on-device latency, energy, model size.

**Milestones.** M1 FP16 baseline + eval + sensitivity map · M2 W8A8 PTQ +
retrieval-aware calibration (target ≤ 0.5 pt) · M3 W4A8 + QAT recovery · M4
on-device report + write-up.

**Risks.** Depthwise activation outliers; attention outliers (keep FP16 / per-head);
INT4 on retrieval likely needs QAT; Core ML op coverage for `Conv1d` + attention.

---

## 2. Plan B — Streaming / real-time contrastive video retrieval

**Goal.** Turn MobileViCLIP from an offline **fixed-8-frame clip** encoder into an
**online streaming** encoder that (a) ingests frames incrementally at real-time
fps with **bounded memory**, (b) emits segment embeddings continuously, and
(c) answers text queries with low latency (moment retrieval over a growing index).

**Why it's open.** 2026 "streaming video" work is generative video-LLM / agents
(StreamSense @ WWW'26, StreamBridge); there is **no on-device streaming
*contrastive* retrieval encoder**.

**Why the current model isn't streaming-ready (repo specifics).**
- Window `T=8` is **hard-coded** in reshapes across `RepMixer` / `MHSA` /
  `AttentionBlock` (e.g. `x.reshape(8, …)` at L597, L1031), and the final
  embedding is a **mean over the whole 8-frame clip** (`FastViT.forward` L1291) —
  both need the full window.
- The temporal RepMixer 1D conv uses **centered padding** (`padding = k//2`,
  L521–529 / L672–680) → **non-causal**, it peeks at future frames.
- The attention stage does **full joint `T×H×W` attention** (L397–423) → O(T)
  recompute per new frame.

### Approach

1. **Causal-ize the temporal branch.** Switch the temporal depthwise `Conv1d`
   from centered to **left-only (causal) padding**, so frame `t` depends only on
   frames `≤ t`. It still folds to a single `Conv1d` → keeps the reparameterization
   / on-device advantage. Surgical change to `RepMixer` (L549–566 build, L672–684
   fold).
2. **Windowed / causal temporal attention.** Replace full joint attention in the
   last stage with a **sliding causal window** (last `W` frames) or a recurrent
   frame-KV cache, so per-frame cost is O(W) not O(t).
3. **Incremental pooling.** Replace the whole-clip mean (L1291) with a running /
   EMA pooled embedding + a ring buffer of the last `W` frame features → emit a
   segment embedding every `stride S` frames without recompute; memory bounded.
4. **Retrieval loop.** Emitted segment embeddings enter a growing ANN index
   (FAISS-flat on device); the text encoder runs once per query; cosine top-k →
   moment localization.
5. **Training.** Fine-tune from pretrained MobileViCLIP with the causal temporal
   branch on `internvid_all` (VTC loss on causal windows). **Add temporal hard
   negatives** (shuffled / reversed windows) so the causal model actually uses
   order — this directly answers the 2025–26 "order-blind" diagnostic thread
   (TVBench / TemporalBench; VideoComp / TempCLR-style negatives). Optionally
   distill from the non-causal FP teacher on overlapping windows.
6. **Latency budget.** Target ≥ 30 fps ingestion + sub-100 ms query on device.
   The spatial path is already per-frame (`FastViT.forward` folds T into batch),
   so the causal temporal state keeps marginal per-frame cost ~constant.

**Novelty.** First on-device streaming contrastive video-text retrieval encoder;
causal *reparameterizable* temporal branch (keeps MobileViCLIP's deploy graph);
incremental bounded-memory pooling; temporal-hard-negative training against
order-blindness.

**Eval.** (a) streaming / long-video moment retrieval (MomentSeeker, LoVR,
ActivityNet); (b) causal-vs-offline penalty on MSR-VTT / DiDeMo R@1; (c) throughput
(fps), query latency, peak memory on device.

**Milestones.** M1 causal temporal conv + incremental pooling, verify parity with
offline on short clips · M2 fine-tune on InternVid causal windows + hard negatives ·
M3 streaming index + real-time query loop · M4 long-video benchmark + on-device
throughput.

**Risks.** Causal penalty (loses future context) — quantify it; attention-windowing
quality; on-device ANN memory growth (cap / segment it); hard-negative training
stability.

---

## Notes

- Line numbers reference `models/mobileviclip_small.py` at the time of writing;
  re-locate by symbol (`reparameterize`, `RepMixer`, `AttentionBlock`,
  `FastViT.forward`) if the file drifts.
- All figures cited above are author-reported (see the report's caveats); the
  point of both plans is to produce **independently measured** on-device numbers,
  which the field currently lacks for video-CLIP.
