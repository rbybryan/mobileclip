#!/usr/bin/env python3
"""C_fuse vote stage: cluster A_llm views across the 5 recaps -> voted units.

Master plan §5.5 (2026-07-27): instead of consuming per-recap views as-is,
cluster the derived short views ACROSS the N independent recaptions of the same
clip; a cluster's support = how many distinct recaps contributed a member.
Units with support >= k (preregistered k*=3, §5.2) are "voted reliable" and go
to the LLM condense call (vlm_audit_hf --mode condense) which packs them into
2-4 dense <=28-char training views. Units below threshold are recorded as
`dropped` -- the voting effect stays visible in the TSV.

Pure python (no model): same-fact matching uses char-3-gram Jaccard, with an
en_gloss overlap rescue for paraphrases that share detected objects but few
character n-grams. All thresholds are CLI-tunable.

Input : derive-output glob (records {"video", "verdict": {"views": [
        {"text", "gloss", "src_recap"}]}})
Output: jsonl {"video", "units": [{text, support, gloss, members}],
        "dropped": [{text, support}], "n_views_in", "n_units", "fallback"}

Example:
    python tools/data_audit/fuse_condense.py \
        --views 'outputs/audit_10k/pipe_views.*.jsonl' \
        --scores 'outputs/audit_10k/pipe_scores.*.jsonl' \
        --output outputs/audit_10k/pipe_units.jsonl
"""

import argparse
import glob
import json
import re
import sys

STRIP = re.compile(r"[\s，。；！？、,.;!?:：\"“”'‘’（）()\[\]【】]+")


def ngrams(text, n=3):
    s = STRIP.sub("", text or "")
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b):
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def same_fact(v1, v2, jac_th, gloss_th, jac_lo):
    """Two views describe the same fact if their 3-gram texts overlap strongly,
    or their detected-object glosses agree (>=2 shared nouns -- one shared noun
    must not merge two different facts about the same object) while the softer
    2-gram overlap clears a low floor (zh paraphrases often share almost no
    3-gram but plenty of 2-grams)."""
    if jaccard(v1["_ng"], v2["_ng"]) >= jac_th:
        return True
    g1 = {g for g in (v1.get("gloss") or [])}
    g2 = {g for g in (v2.get("gloss") or [])}
    return (len(g1 & g2) >= 2 and jaccard(g1, g2) >= gloss_th
            and jaccard(v1["_ng2"], v2["_ng2"]) >= jac_lo)


def cluster_views(views, jac_th=0.35, gloss_th=0.5, jac_lo=0.10,
                  best_recap=None):
    """Union-find clustering -> units sorted by (support desc, len(members) desc).

    Each unit: {text, support, gloss, members}. `support` counts DISTINCT
    src_recap values. Representative text = member from `best_recap` when
    given (SigLIP-top recap), else the most central member (max summed
    Jaccard to the rest), ties -> shortest."""
    vs = []
    for v in views:
        t = (v.get("text") or "").strip()
        if not t:
            continue
        vs.append({**v, "text": t, "_ng": ngrams(t), "_ng2": ngrams(t, 2)})
    parent = list(range(len(vs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if same_fact(vs[i], vs[j], jac_th, gloss_th, jac_lo):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(len(vs)):
        groups.setdefault(find(i), []).append(vs[i])

    units = []
    for members in groups.values():
        support = len({m.get("src_recap") for m in members})
        rep = None
        if best_recap is not None:
            cands = [m for m in members if m.get("src_recap") == best_recap]
            if cands:
                rep = min(cands, key=lambda m: len(m["text"]))
        if rep is None:
            rep = max(members,
                      key=lambda m: (sum(jaccard(m["_ng"], o["_ng"])
                                         for o in members if o is not m),
                                     -len(m["text"])))
        gloss = sorted({g for m in members for g in (m.get("gloss") or [])})
        units.append({"text": rep["text"], "support": support, "gloss": gloss,
                      "members": sorted({m["text"] for m in members})})
    # tie-break by detectable-noun count before text: when the fallback keeps a
    # single unit, a verifiable subject fact must beat an incidental detail
    units.sort(key=lambda u: (-u["support"], -len(u["members"]),
                              -len(u["gloss"]), u["text"]))
    return units


def iglob_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def load_best_recap(scores_glob):
    """{video -> 0-based index of the SigLIP-top recap} from tag=recap_i rows."""
    best = {}
    if not scores_glob:
        return best
    for line in iglob_lines(scores_glob):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag, s = rec.get("tag") or "", rec.get("sig_zh_mean")
        m = re.fullmatch(r"recap_(\d+)", tag)
        if m and s is not None:
            v = rec.get("video")
            if v not in best or s > best[v][0]:
                best[v] = (s, int(m.group(1)) - 1)
    return {v: i for v, (_, i) in best.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", required=True,
                    help="derive-output glob (verdict.views with src_recap)")
    ap.add_argument("--scores", default=None,
                    help="optional score_alignment glob (tag=recap_i) -- picks "
                         "each unit's representative from the SigLIP-top recap")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-support", type=int, default=3,
                    help="keep units seen in >= this many distinct recaps "
                         "(preregistered k*=3, master plan §5.2)")
    ap.add_argument("--jac-th", type=float, default=0.35,
                    help="3-gram Jaccard above which texts merge outright")
    ap.add_argument("--gloss-th", type=float, default=0.5,
                    help="gloss Jaccard for the >=2-shared-noun rescue path")
    ap.add_argument("--jac-lo", type=float, default=0.10,
                    help="2-gram Jaccard floor for the gloss rescue path")
    args = ap.parse_args()

    best = load_best_recap(args.scores)
    n_clips = n_kept = n_drop = n_fallback = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for line in iglob_lines(args.views):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = rec.get("video")
            views = (rec.get("verdict") or {}).get("views") or []
            units = cluster_views(views, args.jac_th, args.gloss_th,
                                  args.jac_lo, best_recap=best.get(vid))
            kept = [u for u in units if u["support"] >= args.min_support]
            fallback = False
            if not kept and units:
                # 保底回退 (mirrors C_v discipline): never leave a clip with
                # zero units -- take the top-voted one and flag it
                kept, fallback = units[:1], True
                n_fallback += 1
            dropped = [{"text": u["text"], "support": u["support"]}
                       for u in units if u not in kept]
            fout.write(json.dumps({
                "video": vid, "units": kept, "dropped": dropped,
                "n_views_in": len(views), "n_units": len(units),
                "fallback": fallback,
            }, ensure_ascii=False) + "\n")
            n_clips += 1
            n_kept += len(kept)
            n_drop += len(dropped)
    print(f"[done] {n_clips} clips -> {n_kept} voted units kept "
          f"({n_kept / max(1, n_clips):.1f}/clip), {n_drop} dropped, "
          f"{n_fallback} fallback -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
