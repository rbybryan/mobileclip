# Top-Tier Paper Plan (v1) — re-cut of the improvement roadmap

**Date:** 2026-07-16
**Goal shift:** the previous notes (`model-architecture-improvements{,-zh}.md`,
`script-level-optimizations-zh.md`) were a *replication + improvement backlog* —
10 proposals + 13 script fixes, all worth doing in the abstract. This document
re-cuts that backlog for a different objective: **one novel, top-tier-conference
paper** (CVPR 2027 main target, deadline ~Nov 2026; ICLR 2027 ~Sept as a stretch,
ECCV'26 window already closed). A paper is a single story with 2–3 contributions
and airtight ablations — not a grab-bag. Everything below is therefore sorted
into **spine / component / skip**.

---

## 1. The story (one sentence)

> **Dataset-Reinforced training with temporal-elastic distillation turns a
> mobile video-CLIP into a single model that matches much larger video-text
> models at any frame budget T ∈ {2…16} — with zero inference overhead.**

Working title direction: *"Video Dataset Reinforcement: Training Elastic
Mobile Video-Text Models"* (name TBD).

Why this spine and not the others we considered:

- **Novelty is defensible.** MobileCLIP/MobileCLIP2's DR recipe has **no
  published video counterpart** (verified in the 2026-07 literature sweep).
  "Apply DR to video" alone would risk an *incremental* verdict — so the spine
  adds a genuinely new axis: **temporal-elastic distillation** (teacher encodes
  T=16 frames once, offline; student trains at random T ∈ {2,4,8,16} against the
  *same* stored teacher embedding). That gives one checkpoint whose frame count
  is a deployment knob, and the claim "a 4-frame mobile model distilled from a
  16-frame teacher beats an 8-frame VTC-only model" is a new, testable result.
  No prior work combines DR-style offline reinforcement with frame-count
  elasticity.
- **It is the highest-expected-gain item we validated** (image-side DR is worth
  +2–5 points at equal student size — P5 was already ranked "largest expected
  accuracy jump").
- **It fits the compute we actually have** (8×V100-32GB): the expensive part is
  one offline teacher pass (embarrassingly parallel, resumable), and student
  training *gets cheaper* than the baseline (teacher does the hard mining;
  MobileCLIP reports fewer steps needed).
- **Everything reuses validated repo assets**: `caption_augmentation` hooks,
  `vtc_loss_sigmoid`, `AttentionPoolingBlock` pattern, `interpolate_pos_embed`.

## 2. Contributions as a reviewer will see them

| # | Contribution | Built from | Novelty level |
|---|--------------|-----------|---------------|
| C1 | **Video Dataset Reinforcement**: offline InternVideo2 teacher (video+text embeddings, real + synthetic recaptioned captions) stored once over InternVid-10M-FLT; student trains with VTC + embedding-KD + similarity-matrix-KD | P5 + `data-quality-caprl-scheme.md` | First for video; recipe-level |
| C2 | **Temporal-elastic distillation**: T-agnostic temporal ops + random-T training against a fixed 16-frame teacher target → one model, frames-as-knob; Pareto curve beats fixed-T baselines at every budget | P2 **merged into** P5 | The new idea — this is what makes it a paper, not a port |
| C3 | **Recipe components with ablations**: sigmoid multi-positive loss (learnable t/b, `get_mask` ported), temporal attention-pool head (zero-init, strictly-no-regression) | P4 + P3 | Known techniques, our ablation table |
| C4 | **Results**: ZS retrieval (MSR-VTT/DiDeMo/ANet/MSVD/LSMDC) + ZS classification (K400, SSv2-MC) at multiple T, **on-device latency table**, vs MobileViCLIP / InternVideo2-S/B / TempMe-class efficient baselines | eval harness | — |

Free bonus folded into C1's loss loop: Matryoshka prefixes (P10) — one
paragraph + one table column, zero extra runs.

**Novelty defense (pre-empt the obvious review):** "isn't this MobileCLIP2 on
video?" — No, because video breaks the image recipe in specific ways we solve:
(a) the teacher's frame count ≠ student's (C2 is the answer, and it's new);
(b) InternVid captions are noisy and *shared across clips of one video* — batch
softmax treats these as false negatives; the sigmoid multi-positive objective
with `idx`-masking (C3) is the video-specific fix; (c) mean-over-frames throws
away the temporal saliency the teacher embedding contains — the attention pool
(C3) is where the distilled temporal signal gets capacity. Each of (a)–(c) gets
its own ablation row.

---

## 3. Disposition of every existing point

### Spine (do, in this order)

| Old # | New role | Change from old plan |
|-------|----------|----------------------|
| P5 | **C1 core** | Upgraded: add synthetic-caption channel from day one (CapRL note); teacher = InternVideo2-L at T=16 (decide L vs 1B on a 100k slice first) |
| P2 | **C2 core** | Upgraded, no longer a standalone "un-hardcode T" chore: T-agnostic reshapes + `temporal_embedding` interpolation are now *part of the method*, and random-T training is the contribution |
| P4 | C3 component | Unchanged (wiring `vtc_loss_sigmoid` + `get_mask` port, SigLIP init t=log 10, b=−10) |
| P3 | C3 component | Unchanged (zero-init attention pool over T frame embeddings) |
| P10 | C1 bonus | Simplified: nested-prefix loss loop only; no separate study |

### Simplify (do the minimum that unblocks the spine)

| Old # | Minimum version | What we drop |
|-------|-----------------|--------------|
| P1 | Bump env to torch ≥2.0 + SDPA (or xformers on 1.13 if DeepSpeed blocks the bump). Infra only, one line in the paper's impl details | Any "efficient attention" claim |
| P9 | Only if the latency table needs it: a divided-attention *deployment variant* fine-tuned with the C1 loss, reported as one ablation row + the Core ML export demo | Full study of divided vs joint; hybrid variants |
| H1/H4 + freeze bug | Mandatory hygiene **before any baseline number**: remove `set_detect_anomaly`, remove spikingjelly, fix `hasattr(name,'_1d')` | — |
| H3 | Mandatory: report `dsl/` and `raw/` scores separately, pin the paper's protocol to one of them **before** the first A/B | — |
| H5 | Optional but cheap: shard eval across ranks — pays for itself across the ~15 A/B runs below | — |

### Skip (cut from this project; keep the notes for later)

| Old # | Why skipped |
|-------|-------------|
| P6 (temporal token merging) | TempMe already published it (ICLR'25); as our contribution it's derivative, and our own analysis showed stage-4 attention is only ~8% of MACs. **Cite and compare against it instead** — TempMe becomes a baseline row, not a work item |
| P7 (asymmetric frame encoding) | Genuine novelty but a *second paper's* worth of risk; C2 already owns the "fewer frames, same accuracy" territory with far lower risk. Revisit as the follow-up if C2 lands (the self-distillation loss from C1 is exactly what P7 needs) |
| P8 (mct text tower) | Pure compression, no novelty; doesn't interact with the story. Future work / appendix at most |
| H2, M1–M3, L1–L5 | Engineering polish with zero paper value; do only if a specific run is blocked by memory/speed |
| whitespace §3 (quantization), §4 (frame selection) | §4 is subsumed by C2 (elastic T is the stronger version of frame selection); §3 stays parked |

whitespace §2 (temporal hard negatives) is **absorbed**: with teacher
embeddings stored per clip, temporal hard negatives (shuffled-frame clips
scored by the teacher) become nearly free — keep it as a *stretch ablation*
inside C1, not a separate phase.

---

## 4. Experiment plan (8×V100, all A/Bs matched steps + data budget)

```
Wk 1      Hygiene gate: H1, H4, freeze bug, H3 protocol pin, torch bump (P1).
          Re-run baseline eval → LOCK baseline numbers (both scoring protocols).
Wk 2      C3 A/Bs (cheap, 3-epoch short runs): sigmoid loss; attn pool; both.
          Kill criterion: keep a component only if ≥ +0.3 R-Sum vs locked baseline.
Wk 2-3    Teacher pass, staged: 100k slice → measure GPU-hrs, storage, and
          *teacher ZS quality on our eval suite* (sanity: teacher must dominate
          student by a wide margin). Decide InternVideo2-L vs 1B. Then full 10M
          (resumable shards; ~1 GPU-wk class — runs while Wk 3 code lands).
Wk 3      C2 code: T-agnostic ops + temporal_embedding interpolation + random-T
          sampler. Unit test: T=8 path bitwise-matches the old hard-coded path.
Wk 4-6    Main runs (the paper's table 1):
            R0 baseline VTC-only, fixed T=8            (reproduced)
            R1 + C3 components                         (best of Wk 2)
            R2 + C1 DR loss (fixed T=8)                → isolates DR gain
            R3 + C2 random-T (full method)             → the headline model
          Each ~3 epochs on InternVid-10M ≈ the original training budget.
Wk 7      Ablations: λ1/λ2, caption mix, T-distribution; MRL column;
          (stretch) temporal hard negatives; (if needed) P9 deployment variant.
Wk 8      On-device latency (Core ML / ONNX at T ∈ {2,4,8,16}), TempMe +
          InternVideo2-S/B baseline table, writing.
Buffer    ~6 wks to the CVPR deadline for reruns and rebuttal-proofing.
```

**Go/no-go gates:**

- After R2: DR gain < +1.0 avg R@1 over R1 → the spine is in danger; fallback =
  pivot the paper to the efficiency story (promote P7 + P9, demote DR to a
  training detail). Decide *then*, not before.
- After R3: elastic model at T=4 must beat fixed-T=4-trained baseline AND be
  within ~1 point of its own T=8 mode — otherwise C2 becomes "we also tried"
  and the paper ships on C1+C3 (weaker but viable, TMLR/AAAI class).

## 5. What changed vs the old sequencing, in one look

- Old 阶段 0–3 treated P1–P10 as parallel tracks; **now there is one track**
  (C1→C2) plus cheap component A/Bs, and the efficiency line (P6/P7/P9) is
  cut or reduced to baselines/one ablation.
- P2 was "engineering prerequisite"; it is now **half of the novelty** (C2).
- P5's synthetic captions moved from "optional channel" to day-one (they are
  MobileCLIP2's single biggest lever and our caption-noise defense).
- Script list: only the 4 correctness items (H1, H3, H4, freeze bug) survive
  as blockers; everything else is opportunistic.

*Line/file references inherit from the two validated notes; all facts cited
here were code- or literature-verified there.*
