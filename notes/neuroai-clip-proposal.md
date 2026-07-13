# Linking Efficient (Video-)CLIP to Human Visual Neural Data
### A NeuroAI project proposal — panel synthesis

*Vehicle repo: this MobileViCLIP replication, which ships a matched, same-family
**image** encoder (Apple MobileCLIP, `mobileclip_s0/s2`) and **video** encoder
(`mobileviclip_tiny/small`). Draft v1, 2026-07-13.*

> **One-paragraph pitch.** Almost all CLIP-vs-brain work uses a *static image*
> CLIP and asks *where semantics live* in ventral cortex. We have something rare
> in-house: a fast **video** CLIP and its **matched image sibling** trained in the
> same family. That lets us cleanly isolate *dynamics* — the representational
> content that only exists once frames are in motion — and ask a question static
> CLIP literature structurally cannot: **where, and when, does motion become
> *meaning* in the human visual system, above and beyond static form and low-level
> motion energy?** The flagship is a controlled variance-partitioning encoding
> study on open dynamic-fMRI data (BOLD Moments, Algonauts-2025), with a
> within-model *temporal-lesion* control that is the methodological crux, plus a
> THINGS-MEG arm for the *when*. Efficiency is not a footnote: it is the
> instrument that makes the full layer × time × ROI × direction search tractable.

---

## 1. Why this repo is the right vehicle

Three properties of this codebase are scientific assets, not just engineering ones:

1. **Matched image + video encoders in one family.** `mobileclip_s2` (image-text)
   and `mobileviclip_small` (video-text) share training lineage and embedding
   geometry. Comparing them isolates *"video vs image"* far more cleanly than
   comparing, say, InternVideo2 against OpenCLIP (which differ in data,
   objective, scale, and architecture all at once).
2. **Efficiency as an experimental enabler.** MobileCLIP's reinforced/distilled
   design gives near-SOTA embeddings at a fraction of the compute. That converts
   otherwise-prohibitive sweeps — every voxel × every layer × every timepoint ×
   thousands of interpretable directions × per-subject refits — into a laptop-
   scale study. It also opens a *class* of experiments others can't run:
   real-time / closed-loop decoding, per-subject test-time alignment, exhaustive
   ablation.
3. **A working feature-extraction + eval pipeline already exists.**
   `models/mobileviclip_small.py`, `models/backbones/`, and `tasks_clip/` already
   load checkpoints and run clip/video-text inference. We repurpose that as a
   brain-encoding feature extractor rather than building from scratch.

---

## 2. Panel perspectives (roundtable)

**👁️ Model / mechanistic-interpretability lens.** SOTA CLIPs diverge along a
*data-curation* axis (MetaCLIP2, DFN) and an *objective/geometry* axis (SigLIP2's
sigmoid loss, better dense features). Wang, Kay, Tarr & Wehbe (*Nat. Mach.
Intell.* 2023) showed **language supervision + data diversity — not architecture —**
is what makes CLIP explain high-level visual cortex, so those axes are the real
levers. Video splits into two representational *species*: **contrastive-semantic**
(InternVideo2, ViCLIP, LanguageBind) and **predictive-latent** (VideoMAE,
V-JEPA2). Mechanistic-interpretability now yields *named, linear* CLIP directions
— TextSpan attention heads (Gandelsman et al., ICLR 2024) and sparse-autoencoder
concept dictionaries — that structurally mirror BrainSCUBA's voxel captions and
could be **tested as brain axes**. Recommendation: use interpretable directions as
encoding bases; the efficiency of MobileCLIP is what makes scoring thousands of
them across every voxel tractable.

**🧠 Neural-data lens.** The open datasets form a clean ladder. **THINGS**
(Hebart 2019) + its **SPoSE** behavioural embedding (49-dim, *Nat. Hum. Behav.*
2020; 66-dim, *eLife* 2023) give an interpretable *ground-truth mental-space*
target. **THINGS-EEG2** (Gifford 2022, 10 subj, 1 kHz, 200-concept×80-repeat test
set) is the de-facto **zero-shot decoding benchmark**; **THINGS-MEG** (Hebart
2023, 4 subj, 271-ch) is the best **millisecond** trajectory of single-image
recognition. **NSD** (Allen 2022, 8 subj, 7T, 73k COCO) is the static-fMRI
workhorse (→ Algonauts 2023). The frontier is **dynamic fMRI**: **BOLD Moments**
(Lahner *Nat. Commun.* 2024, 10 subj, 1000+ 3-s clips — the "video NSD") and
**Algonauts-2025** (CNeuroMod movies, ~65 h, explicit OOD generalization; won by
Meta's **TRIBE** fusing Llama-3.2 + **V-JEPA2** + audio). The key structural gap:
a **temporal-vs-video "scissors"** — M/EEG gives fast dynamics but only to static
images; video-fMRI gives real motion but at second-scale BOLD. Recommendation:
matched image-vs-video variance partitioning on BOLD Moments + Algonauts-2025.

**🔬 Moderator's prior-art / novelty pass** (covering the scoop-check lens). What
is *solidly done*: (a) image-CLIP **encoding** of ventral cortex (Conwell et al.
*Nat. Commun.* 2024 — *scale/data*, not architecture, drives alignment; Doerig et
al. 2024 — semantic scene descriptions predict cortex); (b) static-image
**decoding/reconstruction** on NSD (Takagi & Nishimoto 2023; Brain-Diffuser,
Ozcelik & VanRullen 2023; MindEye, Scotti 2023; **MindEye2**, Scotti 2024,
cross-subject); (c) **M/EEG decoding** of visual perception (Benchetrit, Banville
& King, ICLR 2024; Song et al. contrastive THINGS-EEG decoding); (d) **dynamic
encoding** for leaderboards (TRIBE, Algonauts-2025). What is **not** done, and is
our white space: **nobody has used a matched same-family image/video CLIP pair to
partition dynamic-vs-static neural variance as a scientific claim** (TRIBE *fuses*
modalities to win, but never decomposes what *dynamics per se* contribute, nor
controls for low-level motion energy); and **nobody has used decomposed,
text-labeled CLIP directions as an interpretable encoding basis** validated
against ventral-stream organization. Both are open.

**🧪 Moderator's feasibility / design pass** (covering the skeptic lens). The
claim "video-CLIP beats image-CLIP in motion regions" is *trivially confoundable*
by raw **motion energy** (a dumb optical-flow model also lights up MT+). It is
also confoundable by **architecture/training** differences between MobileCLIP and
MobileViCLIP. The design must therefore neutralize both — see §5. Non-negotiable
baselines: motion-energy/optical-flow, single-frame and frame-averaged image-CLIP,
a vision-only SSL model (DINOv2) and a video SSL model (VideoMAE/V-JEPA2), a
low-level Gabor/GIST model, and category/annotation models. All accuracies
**noise-ceiling-normalized**; strict train/test separation to avoid
double-dipping (Kriegeskorte 2009).

*(Agents 3 and 4 — dedicated prior-art and feasibility — were still running at
synthesis time; their intended coverage is folded into the two moderator passes
above and will be reconciled if they return.)*

---

## 3. The white space (what makes this novel)

| # | Open problem | Why it's still open |
|---|---|---|
| W1 | *Dynamic-beyond-static* cortical variance, cleanly identified | Prior video-fMRI work fuses features for leaderboards; never isolates temporal information with a within-model control + motion-energy partial |
| W2 | *Motion → meaning* vs *motion → motion-energy* | The MT+/STS confound is rarely partialled out, so "semantic dynamics" claims are weak |
| W3 | Interpretable CLIP **directions as brain axes** | BrainSCUBA captions voxels; the inverse — decomposed CLIP directions *predicting* + *labeling* cortex — is untested |
| W4 | Efficient, **open, deployable** brain encoder | SOTA encoders (V-JEPA2-Giant) are heavy; a fast matched family enables exhaustive science + closed-loop |

The flagship attacks **W1 + W2** (highest novelty × feasibility). W3 and W4 are
staged as stretch arms.

---

## 4. Central question & hypotheses

**Question.** *Where in the visual hierarchy, and when in the processing
timeline, does dynamic (video) representational content explain human neural
responses beyond static form and low-level motion — and can we read that content
out in interpretable, text-labeled terms?*

- **H1 — Where.** Video-CLIP (MobileViCLIP) explains reliable fMRI variance
  **beyond frame-averaged image-CLIP** specifically in lateral occipitotemporal
  cortex (LOTC), STS, and dorsal motion areas (MT+/MST); ventral object cortex
  (VTC) variance is largely captured by static form.
- **H1a — Motion → meaning (the decisive control).** That dynamic unique variance
  **survives partialling out a low-level motion-energy model** in LOTC/STS →
  it reflects event/action *semantics*, not raw motion. (If it does *not* survive
  outside MT+, the honest conclusion is "video-CLIP's dynamic gain is largely
  motion energy" — a publishable negative result.)
- **H2 — When.** In THINGS-MEG, CLIP semantic geometry and the 66-dim SPoSE axes
  become decodable with a characteristic latency ordering (low-level/animacy
  early ≈100–150 ms; full CLIP geometry later ≈200–300 ms, with recurrence).
- **H3 — Interpretable axes.** Decomposed, text-labeled CLIP directions
  (TextSpan heads / SAE concepts) predict brain responses ≈ as well as the raw
  embedding, and the significant directions per region reproduce known ventral
  organization (animacy/real-world size; Konkle & Oliva 2012) and align with
  BrainSCUBA voxel captions.

---

## 5. Flagship experiment — *"Where and when motion becomes meaning"*

**Datasets.** BOLD Moments (short clips, *no edit cuts* → cleanest for the core
claim) as primary; Algonauts-2025 movies for an **OOD-generalization** replication
(train on Friends S1–6, test on unseen films). THINGS-MEG for the *when* arm.

**Feature spaces (all from this repo's encoders unless noted).**
- **Video-CLIP:** `mobileviclip_small` on the real, temporally-ordered clip.
- **Static form:** `mobileclip_s2` on frames, then frame-averaged (and, separately,
  single-centre-frame) — the cross-family static control.
- **Temporal-lesion controls (the crux, within one model):** the *same*
  `mobileviclip_small` encoder fed (a) the real clip, (b) a **frame-shuffled**
  clip, (c) a **static-repeated** single frame. The variance explained by (a)
  **minus** (b)/(c) is dynamic information *with architecture and training held
  constant* — this is what neutralizes the MobileCLIP-vs-MobileViCLIP confound.
- **Low-level motion:** a motion-energy / optical-flow model (Nishimoto & Gallant
  2011 style) — the partial for H1a.

**Encoding model.** Per-voxel/parcel **banded ridge regression with variance
partitioning** (Nunez-Elizalde, Huth & Gallant 2019): jointly fit {static-form,
dynamic, motion-energy} bands, read off **unique** and **shared** R² per band.
Nested CV; hemodynamic delay handled by FIR/lagged design.

**Primary outcome.** Noise-ceiling-normalized **unique R² of the dynamic band**,
mapped across cortex, after partialling motion energy. Confirm H1/H1a if it is
significant and localized to LOTC/STS/MT+ and *survives* the motion-energy
partial; falsify if the temporal-lesion contrast (a−b) is ≈0 everywhere or the
surviving signal is confined to early motion areas.

**Baselines / controls (non-negotiable).** DINOv2 (vision-only SSL), VideoMAE or
V-JEPA2 (video SSL, non-contrastive — does the *contrastive/language* part
matter?), Gabor/GIST (low-level), category/annotation model, and a random-network
control. Noise ceilings from stimulus repeats; permutation tests for significance;
FDR across parcels.

**When arm (THINGS-MEG).** Time-resolved RSA + temporal-generalization of
SigLIP2/MobileCLIP embeddings and the 66 SPoSE axes against 271-ch MEG; report
per-dimension emergence latency vs CNN-layer / Gabor / animacy baselines. *Honest
scope:* THINGS-MEG stimuli are static, so this dates the emergence of *semantic
form*, not motion-semantics; a truly dynamic *when* needs fast + naturalistic-video
human data that does not yet exist at scale (this gap is itself a motivating
contribution and a bid for future data collection).

---

## 6. Stretch experiments

- **S1 — Interpretable directions as brain axes (W3).** Decompose MobileCLIP/
  SigLIP2 embeddings into TextSpan heads + SAE concepts; use them as the encoding
  basis; label each significant brain direction with text; validate against
  BrainSCUBA and animacy/size topography. High interpretability payoff.
- **S2 — Efficient open brain encoder + dynamic decoding (W4).** Benchmark
  `mobileviclip_small` vs V-JEPA2/InternVideo2 on Algonauts-2025/BOLD Moments at a
  fraction of the FLOPs; then zero-shot **video-event retrieval** from fMRI via its
  latent (framed scientifically, not as a leaderboard chase).
- **S3 — Real-time / closed-loop (blue-sky, W4).** Exploit on-device latency to
  decode a subject's CLIP-space trajectory live and steer stimulus generation
  along interpretable directions — a genuinely efficiency-*enabled* causal probe.
  Engineering-heavy; park unless a partner lab has streaming M/EEG.

---

## 7. Baselines & controls — quick reference

| Feature space | Role | Guards against |
|---|---|---|
| MobileViCLIP (real clip) | dynamic-semantic | — |
| MobileViCLIP (shuffled / static frame) | **temporal lesion** | arch/training confound |
| MobileCLIP frame-avg / single-frame | static form | "it's just objects" |
| Motion-energy / optical flow | low-level motion | "it's just motion energy" (H1a) |
| DINOv2 | vision-only SSL | "language supervision is irrelevant" |
| VideoMAE / V-JEPA2 | video SSL | "contrastive/language adds nothing" |
| Gabor / GIST / pixels | low-level | early-visual leakage |
| Category / annotation model | labels | "captions alone suffice" |
| Random network | sanity | pipeline artifacts |

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| BOLD's ~1.6 s TR blurs motion into form | Use short no-cut clips (BOLD Moments); the *when* comes from MEG, not fMRI |
| Movie edit cuts / scene changes confound Algonauts arm | Treat Algonauts as OOD *replication*, not primary; model cut events as nuisance regressors |
| Dynamic gain = pure motion energy | H1a motion-energy partial is built into the primary analysis, not an afterthought |
| MobileCLIP≠MobileViCLIP (arch/data) confound | Within-model temporal-lesion control is the primary identification, cross-family is secondary |
| Double-dipping / circular RSA | Strict train/test split, nested CV, permutation nulls (Kriegeskorte 2009) |
| "Everything predicts everything" | Report *unique* variance + noise-ceiling normalization, not raw correlations |
| Citations/venues in this draft unverified | Verify every arXiv ID / venue against the source before any manuscript |

---

## 9. Phased plan (mapped onto this repo)

- **Phase A — Feature extractor.** Wrap `models/mobileviclip_small.py` /
  `mobileclip_s2` to emit per-layer embeddings for arbitrary clips/frames; add the
  frame-shuffle and static-repeat transforms in `dataset/video_utils.py`. Sanity-
  check against the existing `tasks_clip/retrieval.py` zero-shot numbers first.
- **Phase B — Data.** Pull BOLD Moments + THINGS-MEG (open); register Algonauts-
  2025 splits. (Google Drive assets for weights already documented in
  `CHECKPOINTS.md`.)
- **Phase C — Encoding.** Banded-ridge variance partitioning (himalaya); produce
  the dynamic-unique-R² cortical map with the motion-energy partial.
- **Phase D — When.** THINGS-MEG time-resolved RSA against SPoSE + CLIP.
- **Phase E — Write.** Flagship paper; stretch arms S1/S2 as follow-ups.

---

## 10. Publication targets

- **Methods/ML venue** (NeurIPS D&B, ICLR, CCN): the matched-encoder + temporal-
  lesion *method* and the efficient open brain encoder.
- **Neuroscience venue** (*Imaging Neuroscience*, *J. Neurosci.*, *eLife*, *Nat.
  Commun.*): the *where/when motion becomes meaning* result and interpretable
  cortical maps.
- **Bid for impact:** a clean, controlled answer to a real perception question +
  an openly released, deployable brain-encoding model + the interpretable
  direction→axis map. That combination is what separates this from a leaderboard
  entry.

---

## 11. Key references (verify before use)

- Hebart et al. 2019, *PLOS ONE* — THINGS database.
- Hebart et al. 2020, *Nat. Hum. Behav.* — SPoSE 49-dim embedding.
- Hebart et al. 2023, *eLife* — THINGS-data (THINGS-MEG; 66 dims).
- Grootswagers et al. 2022, *Sci. Data* — THINGS-EEG1.
- Gifford et al. 2022, *NeuroImage* — THINGS-EEG2 (zero-shot benchmark).
- Allen et al. 2022, *Nat. Neurosci.* — Natural Scenes Dataset (NSD).
- Chang et al. 2019, *Sci. Data* — BOLD5000.
- Lahner et al. 2024, *Nat. Commun.* — BOLD Moments (video fMRI).
- Gifford et al. 2023 / 2025 — Algonauts 2023 & 2025; TRIBE winner (Meta).
- Wang, Kay, Tarr & Wehbe 2023, *Nat. Mach. Intell.* — language supervision & visual cortex.
- Conwell et al. 2024, *Nat. Commun.* — what drives brain-aligned representation.
- Doerig et al. 2024 — semantic scene descriptions predict cortex.
- Gandelsman, Efros & Steinhardt 2024, ICLR — interpreting CLIP via text-based decomposition.
- Luo, Henderson, Tarr & Wehbe 2024, ICLR — BrainSCUBA.
- Konkle & Oliva 2012, *Neuron*; Konkle & Caramazza 2013, *J. Neurosci.* — animacy/size organization.
- Nunez-Elizalde, Huth & Gallant 2019, *NeuroImage* — banded ridge / variance partitioning.
- Nishimoto & Gallant 2011, *Curr. Biol.* — motion-energy movie reconstruction/encoding.
- Takagi & Nishimoto 2023; Ozcelik & VanRullen 2023 (Brain-Diffuser); Scotti et al. 2023/2024 (MindEye/MindEye2) — NSD decoding.
- Benchetrit, Banville & King 2024, ICLR — MEG decoding of visual perception.
- Zhai et al. 2023 (SigLIP) / Tschannen et al. 2025 (SigLIP2); Xu et al. 2024 (MetaCLIP); Fang et al. 2024 (DFN).
- Vasu et al. 2024, CVPR — MobileCLIP; Yang et al. 2025, ICCV — MobileViCLIP (arXiv:2508.07312).
- Assran et al. 2025 — V-JEPA 2; Tong et al. 2022 — VideoMAE; Wang et al. 2024 — InternVideo2.
