# Top-Tier Paper Plan (v1) — re-cut of the improvement roadmap

**Date:** 2026-07-16
**Revised:** 2026-07-17 — added **R4** (conditional reclaim of P7 asymmetric
frame encoding, dual trigger); **P9** upgraded from "only if needed" to an
explicit week-8 export decision point (it is likely a prerequisite of C4's
latency table).
**Revised 2:** 2026-07-20 — nine corrections from a red-team audit, see
**§6**; where §6 conflicts with earlier text, §6 wins (the C4 benchmark
list and the Wk4–6 run list are rewritten in place).
**Revised 3:** 2026-07-23 — seven plan-level changes from the plan review, see **§7**;
full analysis in `notes/plan-review-upgrades.md` (U1–U10). ⚠️ **§1's novelty claim is now
out of date**: FluxViT (ICCV'25) already does elastic-sampling distillation, so C2's
novelty is narrowed to a four-way combination cell — **§7.1 governs, not §1.**
(Chinese counterpart: [`notes/paper-plan-zh.md`](paper-plan-zh.md) — keep both in sync.)

> ### ⚠️ Precedence, and where §4 was *not* actually rewritten (audited 2026-07-29)
>
> **Order of precedence: §7 > §6 > §1–§5.** Revised-2 above claims "the kill criteria are
> rewritten in place"; **they were not**, and neither were several §4 items that §6/§7 later
> overrode. Until §4 is rewritten, these seven conflicts resolve in favour of the later
> section — each is also flagged inline where it occurs:
>
> | Conflict | §4 still says | Governing text |
> |---|---|---|
> | Kill criterion | Wk2 "≥ +0.3 R-Sum vs locked baseline" | **§6.7** — single-benchmark +0.3 is below seed noise; use multi-benchmark aggregated R-Sum, both protocols, second seed when borderline |
> | P1 torch bump | a Wk1 task | **§6.8** — demoted to conditional; Wk1 keeps only the 4 correctness items |
> | Epoch budget | "~3 epochs" for every main run | **§7.6/U6** — ranking arms 1 epoch; only R0 + winner get 3 |
> | Hardware | 8×V100-32GB | **§7.5/U5** — the real run is 4×H100; and see the wall-clock note below |
> | TempMe | in the C4 zero-shot table (Wk8) | **§7.1** — fine-tuned protocol, **out** of the zero-shot table; separate fine-tune/latency-only block |
> | P9 export | first attempt at week 8 | **§7.2/U1** — conversion smoke immediately; Wk8 keeps only real-device latency. **Since done: ONNX + Core ML pass at all T ⇒ P9 does not trigger** |
> | Novelty | §1 "no prior work combines DR with frame elasticity" | **§7.1** — FluxViT does; never claim "elastic distillation" as new |
>
> **Two further facts postdate every section here** and are tracked in `experiments/ROADMAP.md`
> (its 状态速览 + 方案完整性自查 are the live source of truth, not this file):
> **(1)** measured wall-clock is **9.18 s/it at global batch 512 ⇒ ≈53 h/epoch on full 10M**,
> so U6's "宽裕档 = 33E" is ≈73 days and **infeasible**; the maintainer has exactly 4 H100s
> (no more), making ~6.6 days/R0 a hard constraint. **(2)** MSR-VTT 1k currently **cannot rank
> arms**: 10× more data moved raw r_mean by only +0.82, inside the ±1.6 band, so paired
> query-level bootstrap CI (U10-1) is a **prerequisite**, not a nicety.

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
  ~~No prior work combines DR-style offline reinforcement with frame-count
  elasticity.~~ ⚠️ **OUT OF DATE — §7.1 governs.** FluxViT (ICCV'25,
  arXiv:2503.14237) already does elastic-sampling distillation with multi-budget
  zero-shot retrieval. **Never claim "elastic distillation" as new.** C2's
  surviving novelty is the narrower four-way combination cell: frames-as-knob ×
  mobile backbone × fixed-T teacher anchor × on-device latency.
- **It is the highest-expected-gain item we validated** (image-side DR is worth
  +2–5 points at equal student size — P5 was already ranked "largest expected
  accuracy jump").
- **It fits the compute we actually have** (~~8×V100-32GB~~ → **4×H100**, §7.5/U5;
  and the measured 53 h/epoch makes the budget the binding constraint — see the
  precedence box above): the expensive part is
  one offline teacher pass (embarrassingly parallel, resumable), and student
  training *gets cheaper* than the baseline (teacher does the hard mining;
  MobileCLIP reports fewer steps needed).
- **Everything reuses validated repo assets**: `caption_augmentation` hooks,
  `vtc_loss_sigmoid`, `AttentionPoolingBlock` pattern, `interpolate_pos_embed`.

## 2. Contributions as a reviewer will see them

| # | Contribution | Built from | Novelty level |
|---|--------------|-----------|---------------|
| C1 | **Video Dataset Reinforcement**: offline InternVideo2 teacher (video+text embeddings, real + synthetic recaptioned captions) stored once over InternVid-10M-FLT; student trains with VTC + embedding-KD + similarity-matrix-KD | P5 + `data-quality-caprl-scheme.md` | First for video; recipe-level |
| C2 | **Temporal-elastic distillation**: T-agnostic temporal ops + random-T training against a fixed 16-frame teacher target → one model, frames-as-knob; Pareto curve beats fixed-T baselines at every budget ⚠️ *three different success bars for this claim coexist — "beats at every budget" here, "within ~1 pt of own T=8" in the §4 R3 gate, and U10's pre-registered **non-inferiority** δ=0.3 per T. **U10's is the pre-registered one**; and per ROADMAP G2, δ=0.3 is ~8× below the measured noise floor, so the margin itself is unsettled (`plan-review-upgrades.md` §6.1 lists the three ways out — none chosen yet)* | P2 **merged into** P5 | The new idea — this is what makes it a paper, not a port. **Narrowed by FluxViT — see §7.1** |
| C3 | **Recipe components with ablations**: sigmoid multi-positive loss (learnable t/b, `get_mask` ported), temporal attention-pool head (zero-init, strictly-no-regression) | P4 + P3 | Known techniques, our ablation table |
| C4 | **Results**: ZS retrieval (MSR-VTT/DiDeMo/ANet) + ZS classification & multi-choice (K400/UCF101/HMDB51/~~SSv2-MC~~ — **SSv2-MC data is absent and the order-sensitive suite is shelved, see `notes/order-sensitive-eval.md`**) at multiple T, **on-device latency table**, vs MobileViCLIP / InternVideo2-S/B / **FluxViT-S (GFLOPs-matched row, §7.1)** / ~~TempMe~~ (**TempMe is fine-tuned-protocol: separate block, NOT the zero-shot table**). (07-20: MSVD/LSMDC removed from the committed list — no annotations/configs in the repo; re-add only if built) | eval harness | — |

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
| P9 | **Explicit week-8 decision point** (upgraded 07-17): first try exporting the full model with all temporal ops (Core ML/TFLite/ONNX); if every route fails, this variant becomes a **prerequisite** of C4's latency table, not an option (whitespace §0 verified joint attention is the export blocker). Same shape as before: divided-attention deployment variant fine-tuned with the C1 loss, one ablation row + export demo | Full study of divided vs joint; hybrid variants |
| H1/H4 + freeze bug | Mandatory hygiene **before any baseline number**: remove `set_detect_anomaly`, remove spikingjelly, fix `hasattr(name,'_1d')` | — |
| H3 | Mandatory: report `dsl/` and `raw/` scores separately, pin the paper's protocol to one of them **before** the first A/B | — |
| H5 | Optional but cheap: shard eval across ranks — pays for itself across the ~15 A/B runs below | — |

### Skip (cut from this project; keep the notes for later)

| Old # | Why skipped |
|-------|-------------|
| P6 (temporal token merging) | TempMe already published it (ICLR'25); as our contribution it's derivative, and our own analysis showed stage-4 attention is only ~8% of MACs. **Cite and compare against it instead** — TempMe becomes a baseline row, not a work item |
| P8 (mct text tower) | Pure compression, no novelty; doesn't interact with the story. Future work / appendix at most |
| H2, M1–M3, L1–L5 | Engineering polish with zero paper value; do only if a specific run is blocked by memory/speed |
| whitespace §3 (quantization), §4 (frame selection) | §4 is subsumed by C2 (elastic T is the stronger version of frame selection); §3 stays parked |

### Conditional stretch (added 07-17: reclaimed on trigger, costs no mainline schedule)

| Old # | New role | Reclaim logic and triggers |
|-------|----------|----------------------------|
| P7 (asymmetric frame encoding) | **R4 (optional run)** | It was cut as "a second paper's worth of risk" — but that was the train-from-scratch estimate. **Once R2/R3 exist, the C1 distillation loss and teacher embeddings are already in hand, and P7 collapses to one self-distillation fine-tune** (teacher = the model's own full-frame path), roughly a week. It also complements rather than duplicates C2: C2 saves compute by *seeing fewer frames*; P7 sees every frame but routes most through a cheap path, attacking the ~92%-of-MACs per-frame conv cost and improving the Pareto **at fixed T**. **Dual trigger (both required)**: (a) ≥2 weeks of buffer left after R3; (b) the elastic model degrades visibly at low T (frame information demonstrably matters). **Verdict rule**: must beat elastic T=4 at equal FLOPs — win → a full paper section (combined elastic-T × asymmetric-compute Pareto surface); lose → back to future work, no second attempt |

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
          [!] SUPERSEDED BY §6.7 — +0.3 on one benchmark is below seed noise.
              Use aggregated multi-benchmark R-Sum, both protocols, 2nd seed if
              borderline. Also see ROADMAP G4: paired bootstrap CI is required
              before ANY arm comparison is readable at all.
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
            R3b fixed-T=4 baseline (added 07-20)       → the control for C2's
                 gate; reuses C2's T-agnostic code with a fixed 4-frame
                 sampler, runs alongside R3
          Each ~3 epochs on InternVid-10M ≈ the original training budget;
          (07-20) measure single-epoch wall-clock on day 1 of R0 and, if over
          budget, cut in priority order R0 > R2 > R3 > R3b > R1 (R1 may
          demote to short-run evidence).
          [!] SUPERSEDED BY §7.6/U6 — ranking arms get 1 epoch; only R0 and the
              winner get 3. Wall-clock is now MEASURED, not pending: 9.18 s/it
              @ global batch 512 => ~53 h/epoch on 10M => ~6.6 days per 3-epoch
              R0 on the maintainer's 4 H100s (there is no 5th card). U6's
              "generous tier = 33E" would be ~73 days: not available. Execute
              the normal tier and accept serial queueing.
Wk 7      Ablations: λ1/λ2, caption mix, T-distribution; MRL column;
          (stretch) temporal hard negatives; (if needed) P9 deployment variant.
Wk 8      On-device latency (Core ML / ONNX at T ∈ {2,4,8,16}), TempMe +
          InternVideo2-S/B baseline table, writing.
          [!] AMENDED BY §7.1 — TempMe is a FINE-TUNED protocol and must not sit
              in the zero-shot table; give it a separate fine-tune/latency block.
          [!] AMENDED BY §7.2/U1 — export conversion is no longer a Wk8 item: it
              RAN on 07-27 and ONNX + Core ML pass at every T, so P9 does not
              trigger. Wk8 keeps only real-device latency. Open: TFLite (blocked
              on torch>=2.6) and whether dynamic_axes gives one graph for many T.
Buffer    ~6 wks to the CVPR deadline for reruns and rebuttal-proofing.
```

**Go/no-go gates:**

- After R2: DR gain < +1.0 avg R@1 over R1 → the spine is in danger; fallback =
  pivot the paper to the efficiency story (promote P7 + P9, demote DR to a
  training detail). Decide *then*, not before.
- After R3: elastic model at T=4 must beat fixed-T=4-trained baseline AND be
  within ~1 point of its own T=8 mode — otherwise C2 becomes "we also tried"
  and the paper ships on C1+C3 (weaker but viable, TMLR/AAAI class).

**R4 (optional, added 07-17) — conditional reclaim of P7:**

- Triggers (**both** must hold): (a) ≥2 weeks of buffer after R3;
  (b) R3's elastic model degrades visibly at low T (T=2/4).
- Shape: a ~one-week self-distillation fine-tune — K=2–4 keyframes through
  the full path, remaining frames through the shallow path injected via the
  temporal RepMixer; teacher = the model's own full-frame path, reusing the
  C1 loss. No from-scratch training.
- Verdict rule: beat elastic T=4 at equal FLOPs. Win → new paper section
  (combined elastic-T × asymmetric-compute Pareto); lose → future work.

**Week-8 export decision point (upgraded 07-17, formerly P9's "if needed"):**
first try exporting the complete model with all temporal ops
(Core ML / TFLite / ONNX); if any route works, skip P9 entirely; if all
fail, the P9 deployment variant immediately becomes a **prerequisite task**
for C4's latency table (divided-attention variant fine-tuned with the C1
loss, one ablation row + export demo).

## 5. What changed vs the old sequencing, in one look

- Old 阶段 0–3 treated P1–P10 as parallel tracks; **now there is one track**
  (C1→C2) plus cheap component A/Bs, and the efficiency line (P6/P7/P9) is
  cut or reduced to baselines/one ablation.
- P2 was "engineering prerequisite"; it is now **half of the novelty** (C2).
- P5's synthetic captions moved from "optional channel" to day-one (they are
  MobileCLIP2's single biggest lever and our caption-noise defense).
- Script list: only the 4 correctness items (H1, H3, H4, freeze bug) survive
  as blockers; everything else is opportunistic.
- (07-17 revision) P7 is no longer a plain cut: it returns as **R4** behind a
  dual trigger (post-R2/R3 it is just one self-distillation fine-tune);
  **P9** upgraded from "if needed" to an explicit week-8 export decision
  point — if direct export fails, it is a prerequisite of the latency table.

## 6. Audit corrections (2026-07-20 red-team pass)

An adversarial audit of the plan, with key assumptions re-verified against the
code. Nine findings and their dispositions:

**Fixed in code (`exp/opt-vs-baseline` branch):**

1. **Multi-positive handling was a no-op**: `pt_dataset.__getitem__` returns a
   per-sample-unique `idx` (verified in code), so `get_mask(idx)` degenerates
   to the identity — shared captions stayed false negatives. Fix: corpus-level
   `caption_idx` switch (idx = md5 hash of the caption, stable across ranks),
   enabled by default in the R1a/R1 configs. Novelty-defense point (b) only
   holds from here on.
2. **t/b learning-rate trap**: `logit_scale`/`logit_bias` under the main
   lr=1e-5 barely move from the SigLIP init (b=−10) — effectively fixed t/b.
   Fix: `different_lr` (1e-3) enabled in the R1a/R1 configs.
3. **H2 pulled forward**: `with_cp` is now wired to the
   `gradient_checkpointing` config (default True = original behavior).
   Disabling it where memory allows saves 20–40% backward time on every main
   run — schedule value outweighs its "zero paper value" label.

**Plan amendments (already rewritten in place above):**

4. **Primary protocol = raw cosine**: DSL uses the whole test-set query
   distribution — reviewers treat it as a test-time prior incompatible with
   "zero-shot" claims. DSL survives only in an appendix row for aligning with
   the original paper's numbers.
5. **Run-budget addition**: R3b (fixed-T=4 baseline, the control for C2's
   gate) added to Wk4–6; measure single-epoch wall-clock on R0 day 1 and cut
   R0 > R2 > R3 > R3b > R1 if over budget.
6. **C4 benchmarks trimmed**: MSVD/LSMDC have no repo support; removed from
   the committed list.
7. **Kill criteria tightened**: a single-benchmark +0.3 R-Sum is below seed
   noise; now multi-benchmark aggregated R-Sum, same direction in both
   protocols, second seed for borderline (<0.5) calls.
8. **P1 demoted**: torch≥2.0 bump moves from Wk-1 default to conditional
   (naive attention at 512 tokens runs today; DeepSpeed-compat validation
   risks more schedule than it saves). Wk 1 keeps only the 4 correctness items.
9. **Three C1 design details added to R2**: teacher embedding dim ≠ 512 needs
   a projection scheme; the offline pass must record teacher frame indices and
   the sampling protocol (reproducibility); low-T reachability of the 16-frame
   teacher target (per-T weighting / curriculum KD) is an explicit design
   decision. Also: R2 gains a planned ablation "unfreeze the last N text
   layers" (under LiT freezing, text-KD can only train the projection);
   before submission, re-sweep the literature for the "first video DR" claim
   and check InternVid / InternVideo2 licenses for releasing derived
   embeddings (decides whether "dataset release" can be a contribution).

---

## 7. Plan-review patch (2026-07-23; full analysis in `notes/plan-review-upgrades.md`)

A full review of the plan (incl. the #197 neural-inspired additions) plus a 2024–2026
literature sweep. The skeleton stands; seven plan-level changes override conflicting
text above (mirrors §七 of `paper-plan-zh.md`):

1. **C2 novelty narrowed (the big one).** **FluxViT** (ICCV 2025, arXiv:2503.14237)
   already does elastic sampling (4–24 frames) + token selection, distilled from
   InternVideo2-1B, with single-checkpoint multi-budget **zero-shot retrieval**. C2's
   claimable cell shrinks to: *frame-count as the knob* (no dense decode + selector on
   device) × *mobile conv-hybrid dual encoder* (no token stream to select) × *frozen
   fixed-T=16 anchor-target distillation* (cross-T consistency) × *measured on-device
   latency* (FluxViT reports GFLOPs only). FluxViT-S joins the C4 table; never claim
   "elastic distillation" itself as new (FlexiViT/FluxViT own it). C1 must cite
   OpenGVLab's released InternVideo2-S/B/L + MobileCLIP-backbone VideoCLIP distillations
   and pin its novelty to the *offline reinforced-data recipe*. TempMe is a fine-tuned
   protocol → out of the zero-shot table (separate fine-tuned / latency-only rows).
2. **P9 split; conversion smoke test now (U1).** Op-coverage/conversion needs only
   random weights — do it immediately; Wk 8 keeps only real-device latency.
3. **Per-T BN handling for C2 (U3).** BN running stats drift with T (FFN's temporal
   frequency deviation; code-level evidence in the review §2.2). Default: post-training
   per-T BN recalibration (forward-only, 4 buffer sets); escalate to per-T private
   affine if the drift diagnostic is large. R3 criteria use the *calibrated* protocol;
   phrase as "one trained weight set + per-T calibration buffers".
4. **Graded fallback for low-T degradation (U4):** stratified T sampling + low-T KD
   weighting (§6.9) → sandwich sampling (US-Nets, ≈1.6× step budget) → only then R4.
5. **Global batch alignment (U5):** 4×H100 default 256 ≠ upstream 512; prefer 4×128=512
   for R0 fidelity (validate memory at first light); arms never mixed.
6. **Budget ladder (U6):** after day-1 epoch wall-clock E: normal tier = ranking arms
   (R1×3, R2+KD sub-arms, R3, R3b, A1/A2) at 1 epoch matched steps; R0 + winner at full
   3 epochs. The run matrix is already ≈33E without a ladder.
7. **Statistics (U10):** query-level bootstrap 95% CIs in `collect_results.py`; two
   seeds for the winner; pre-registered per-T non-inferiority margin δ=0.3 R@1.

Also: §6.9's pre-submission re-sweep expands from "first video DR" to "+ the elastic-
inference axis + FluxViT successors"; student init switches to MobileCLIP2-S2
(HF-verified drop-in; attribution scheme in review U8).

---

*Line/file references inherit from the two validated notes; all facts cited
here were code- or literature-verified there. §7's literature verdicts are as of the
2026-07-23 web sweep.*
