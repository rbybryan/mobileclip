#!/usr/bin/env python3
"""Stage-2 VLM audit: judge sampled clips with a video-capable open VLM.

Talks to any OpenAI-compatible endpoint (works with `vllm serve` for
Qwen3-VL-*, InternVL3.5-*, GLM-4.5V, MiniCPM-V-4.5, Keye-VL...), sends N
uniformly sampled frames plus the en/zh captions, and asks for a structured
JSON verdict covering the four failure axes:

  video<->en alignment | translation quality | granularity | temporal order

plus a suggested action (keep/fix_text/retranslate/recaption/drop) and K
candidate replacement zh captions (<=30 chars each, so they fit the 32-token
text window) -- the audit doubles as a repair-caption source.

Serve first, e.g.:
    vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --limit-mm-per-prompt image=8
Then:
    python tools/data_audit/vlm_audit.py \
        --input outputs/audit/s2_sample.jsonl --video-root /path/to/clips \
        --api-base http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct \
        --frames 8 --concurrency 32 --output outputs/audit/s2_verdicts.jsonl

Deps: requests, pillow, numpy + one video backend (decord | opencv-python | av).
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

PROMPT = """你是严格的视频-文本数据质检员。给你一个视频的 {n_frames} 帧（按时间顺序均匀采样）和两条描述：

英文原始 caption（上游 InternVid 自动生成，可能有错）：
{en}

中文翻译 caption（由 LLM 翻译自上述英文，可能有翻译错误或残留痕迹）：
{zh}

请只依据画面判断，输出一个 JSON 对象（不要输出任何其他文字），字段如下：
{{
  "video_readable": true/false,            // 帧是否可辨认（黑屏/花屏/纯色为 false）
  "en_score": 0-5,                         // 英文 caption 与画面的符合度；0=完全无关, 3=主体对但有错误细节, 5=准确
  "en_issues": [...],                      // 从 ["wrong_subject","wrong_action","wrong_scene","hallucinated_detail","ocr_or_watermark_read","too_generic","temporal_wrong"] 中选，可为空
  "zh_score": 0-5,                         // 中文 caption 与画面的符合度（独立于英文打分）
  "trans_score": 0-5,                      // 仅比较中英两条文本：翻译忠实+流畅度；5=忠实流畅
  "trans_issues": [...],                   // 从 ["mistranslation","untranslated_fragment","artifact_text","awkward_chinese","omission","addition"] 中选，可为空
  "granularity": "coarse"/"adequate"/"dense",  // caption 信息量：coarse=只有笼统一句(如"一个人在说话")
  "temporal_order_relevant": true/false,   // 视频是否包含多个先后事件
  "temporal_order_correct": true/false/null, // 若 caption 描述了事件顺序，顺序是否与画面一致；未描述则 null
  "action": "keep"/"fix_text"/"retranslate"/"recaption"/"drop",
      // keep: 中文 caption 可直接用于训练
      // fix_text: 只需规则清洗（去引号/前缀等）
      // retranslate: 英文没问题但翻译坏了
      // recaption: 英文本身与画面不符，需按画面重写
      // drop: 视频不可用或画面无法配文
  "better_zh": ["...", "..."]              // 2-3 条基于画面的中文短 caption，每条≤30字、检索风格、
                                           // 描述不同侧面（主体动作/场景/时序），画面不清则给 []
}}"""

VERDICT_KEYS = {"video_readable", "en_score", "zh_score", "trans_score", "action"}

# ---------------------------------------------------------------------------
# Text-only translation rubric (--mode trans).
#
# WHY THIS EXISTS. The combined PROMPT above asks for en_score / zh_score /
# trans_score in one JSON, and the 10k run showed the model does not actually
# factorise them: en==trans on 92.5% of records, zh==trans on 98.9%, and
# `action` turned out to be a fixed ladder on that single number
# (5->keep, 4->fix_text, 3->retranslate, <=2->recaption/drop). "Did the
# translation break" was therefore never measured -- see DATA_ISSUES.md
# problem 8 and notes/audit-10k-findings.md conclusion 8.
#
# Two deliberate design choices here, both aimed at that failure:
#   1. NO FRAMES. Translation fidelity is a text-vs-text property; the rubric
#      always said so ("仅比较中英两条文本"). Withholding the video makes the
#      anchor physically unavailable rather than merely discouraged -- and it
#      makes the pass much cheaper, since video decode was the S2 bottleneck.
#   2. EVIDENCE BEFORE SCORE. The model must enumerate concrete mismatches
#      first and is then bound to a scoring rule keyed to that list, so a score
#      cannot be emitted as a free-floating impression of overall quality.
TRANS_PROMPT = """你是严格的中英翻译质检员。下面给你一条英文 caption 和它的中文译文。

**你看不到视频，也不需要看。** 本次只判断一件事：中文是否忠实、通顺地翻译了这段英文。
⚠️ 英文本身写得对不对、符不符合视频画面，**完全不在本次评估范围内**——
哪怕英文描述是错的，只要中文忠实地翻出了它，翻译质量就是好的。

英文原文：
{en}

中文译文：
{zh}

**先列证据，再打分。** 输出一个 JSON 对象（不要输出任何其他文字）：
{{
  "mismatches": [...],      // 逐条列出不忠实之处，每条形如 "英文的 A 被译成了 B"；完全忠实则给 []
  "untranslated": true/false,  // 中文里是否残留成片的未翻译英文
  "trans_issues": [...],    // 从 ["mistranslation","untranslated_fragment","artifact_text","awkward_chinese","omission","addition"] 中选，可为空
  "trans_score": 0-5        // 见下方评分规则
}}

评分规则（**只依据 mismatches 列表，不得凭整体印象**）：
- mismatches 为空且中文通顺 → **必须给 5**
- 只有生硬/翻译腔，意思无损 → 4
- 有 1 处轻微偏差或漏译细节 → 3
- 有明确误译、漏译主要信息、或成片未翻译 → 0-2（越严重越低）"""

TRANS_KEYS = {"trans_score", "mismatches"}

# ---------------------------------------------------------------------------
# Text-only granularity rubric (--mode gran).
#
# WHY THIS EXISTS. The combined rubric's three-way granularity label came back
# coarse 93.6% / adequate 6.4% / dense 0. A count of exactly zero for one of
# three categories is the signature of label-space collapse, not of the data:
# the same run's trans_score only ever used {0,2,3,5} -- 1 and 4 never appeared.
# If `dense` was simply unreachable, 93.6% is an overestimate of "coarse".
#
# The fix is to ask for a COUNT rather than a band. A count is a continuous
# quantity, so it cannot collapse onto a label the way a 3-way choice can, and
# the bands can be re-derived afterwards from whatever cut we want. No frames
# either: how much information a sentence carries is a property of the sentence.
GRAN_PROMPT = """你是中文视频字幕的信息量分析员。下面给你一条中文字幕。

**你看不到视频，也不需要看。** 只分析这句话本身**携带了多少可检索的信息点**。

中文字幕：
{zh}

「信息点」= 一个能用来检索、且能与别的视频区分开的具体要素。分五类清点：
- 主体（谁/什么东西，含数量、属性：「两个穿红衣的女人」记 1 个主体 + 2 个属性）
- 动作（在做什么）
- 场景（在哪里、什么环境）
- 属性（颜色、材质、大小、表情等修饰）
- 时序（先…后…、然后、接着 —— 只有明确表达先后关系才算）

**先逐项列出，再计数。** 输出一个 JSON 对象（不要输出任何其他文字）：
{{
  "subjects": [...],     // 逐个列出，没有则 []
  "actions": [...],
  "scenes": [...],
  "attributes": [...],
  "temporal": [...],     // 明确的先后关系表述
  "n_info": 0            // 上面五个列表的元素总数
}}

注意：「一个男人在说话」= 主体 1 + 动作 1 = `n_info` 2，属于典型的低信息量字幕。"""

GRAN_KEYS = {"n_info", "subjects", "actions"}

# ---------------------------------------------------------------------------
# Text-only caption merge rubric (--mode merge).
#
# WHY THIS EXISTS. `better_zh` gives 2-3 candidates that the PROMPT deliberately
# asked to cover DIFFERENT facets ("描述不同侧面：主体动作/场景/时序"). They are
# complementary, not redundant. A human review found them accurate with few
# hallucinations, so combining them into one dense caption directly attacks the
# thin-caption problem (median 3 info points; problem 14) -- without any new
# visual claim, so no new hallucination risk. Text-only: it only reorganises
# already-verified text, never looks at the video.
MERGE_PROMPT = """你是中文视频字幕编辑。下面是针对**同一个视频**、由质检模型生成的
2-3 条候选字幕，每条各描述画面的不同侧面（主体动作／场景／时序）。

候选：
{cands}

把它们合并成**一条**通顺、信息完整的中文字幕：
- **只能用候选里已有的信息，绝不新增任何候选中没有的细节**（这条最重要，防幻觉）；
- 去重，按「主体＋动作＋场景（＋时序）」自然组织成一句话；
- 若候选之间冲突（同一属性给了不同值），保留更具体的那个，别硬凑两个都写；
- 长度 ≤50 字，检索风格，不要引号／编号／解释。

只输出一个 JSON 对象（不要输出任何其他文字）：
{{"merged_zh": "合并后的中文字幕"}}"""

MERGE_KEYS = {"merged_zh"}

# ---------------------------------------------------------------------------
# Five-sample recaption prompt (--mode recap in vlm_audit_hf.py).
#
# WHY THIS EXISTS. The RECAPTION bucket needs a caption written FROM the frames,
# not translated from the broken English source. The fusion scheme
# (notes/caption-fusion-scheme.md) then decomposes several such captions into
# atomic units and trusts a unit by cross-sample AGREEMENT (support voting).
# That voting is a self-consistency estimator (SelfCheckGPT, arXiv 2303.08896):
# it is only valid if the captions are INDEPENDENT samples of the SAME content.
# So this prompt is deliberately NOT faceted (unlike better_zh in PROMPT, whose
# candidates cover different facets by design and therefore cannot be voted on):
# every call asks for the same comprehensive description, and the front-end runs
# it N times independently at temperature with per-sample frame jitter. Diversity
# is meant to come from sampling variance, not from steering each sample to a
# different facet -- that is what keeps the support count meaningful.
#
# The rules encode the guardrails the design doc calls for: visible-only / no
# off-frame speculation (systematic-hallucination surface), event-order wording
# with NO timestamps (the model sees 8 frames, has no timecodes), verbatim-or-omit
# for on-screen text (OCR is a VLM weak spot -- forbid "completing" partial text),
# and graceful degradation to vague quantifiers instead of confident wrong counts.
# Single dense paragraph so the N outputs are homogeneous for downstream
# atomic-unit decomposition and cross-sample alignment.
RECAP_PROMPT = """你是一个严谨的视频画面描述助手。下面是从同一段视频中按时间先后均匀抽取的若干帧画面。
请把这些画面综合成一段用于训练的中文描述，尽可能覆盖所有真实可见的信息，同时绝不编造。

严格遵守以下规则：
1. 只描述画面中清晰可见的内容。看不清、不确定、被遮挡或需要推测的内容一律省略，不要猜测。
2. 不要描述画面之外的信息：不要推断拍摄地点、时间、人物身份、情绪动机、事件的前因后果，也不要脑补画面外正在发生的事。
3. 按动作在画面中出现的先后顺序来叙述，用“先”“接着”“随后”“最后”等词表示顺序。禁止出现帧号、秒数、时间戳或“第几张图/第几帧/视频/画面/图片”这类元描述词。
4. 屏幕上的文字（字幕、标题、标牌、水印等）：只有当文字清晰可辨时，才用中文引号“”原样照抄；不要翻译、不要补全、不要猜测残缺的文字；无法辨认时直接不提这段文字。
5. 数量：只有在数量很少且能明确数清时才给出具体数字；否则用“几”“一些”“一群”“许多”等模糊量词，绝不编造精确数目。
6. 覆盖以下方面（只写真实可见的）：主体（人/物的外观、服装、颜色等特征）、主体的动作及其先后、场景与背景环境、屏幕上的文字。如果某一方面在画面中不存在，就完全不要提它。
7. 输出一段连贯、信息密集的中文，不要分点、不要标题、不要开场白或结束语，只输出这段描述本身。"""

# --mode fuse-direct (§5.7 v5, maintainer 2026-07-28): TWO text-only calls
# per clip, decoupled per maintainer ("短稿长稿可以分开生成；长稿不用字数
# 限制"). Call 1 (FUSE_LONG) reads the N independent recaptions and performs
# EDITORIAL JUDGMENT — find the core content, weigh each detail
# (cross-version agreement is evidence, not a hard rule: hard-to-fabricate
# single-source details may stay; easy-to-hallucinate types — numbers,
# on-screen text, names, intent — are dropped or blurred without
# corroboration; contradictions take the majority or get blurred) — then
# writes ONE detail-preserving long draft with NO length cap + the
# reason-annotated dropped list. Call 2 (FUSE_SHORT) sees ONLY the long
# draft (closed set — judgment already happened) and distills THE training
# caption: one natural sentence <=30 chars (32-token CLIP text-encoder
# window: [CLS] + ~30 Chinese chars + [SEP]). Auditability: dropped
# reasons on call 1, rule vote as diagnostic, Panel-V verify+refine on the
# short caption as the hard gate. Both consumed via .format -> doubled
# braces.
FUSE_LONG_KEYS = {"long"}
FUSE_LONG_PROMPT = """下面是同一段视频的 {n} 段互相独立的中文描述。它们出自对同一段视频的多次独立观看，各自都可能有错漏；多段一致的信息通常更可信，但一致与否只是证据、不是铁律——请你像资深字幕编辑一样综合判断。

请在心里分三步完成编辑，最后只输出结果 JSON：

第一步·抓重点：先确定这段视频的核心内容——主要的人/物是谁、正在发生什么。

第二步·辨真伪，对每个细节做取舍：
- 多段独立提到且不矛盾 → 可信，保留；
- 各段互相矛盾（颜色、数量、动作不一致）→ 取多数说法；没有多数就舍弃，或改用更模糊但稳妥的说法（如把存疑的"两名男子"写成"男子"）；
- 只有一段提到 → 看类型：具体的动作、位置这类"难凭空编造"的细节可酌情保留；数字、画面文字内容、品牌/人名、身份、情绪意图这类"易幻觉"的细节一律舍弃或模糊化；
- 引号“”内的画面文字：多段一致才保留且必须原样照抄、不改一字；只有一段提到或各段不一致，就完全不写这段文字。

第三步·成长稿：把**全部**通过取舍的信息写成一段连贯自然的中文描述，按"主体→动作先后→场景→画面文字"组织。**长度不设限制**——以完整覆盖保留信息为准，宁全勿缺；但只写你判定保留的信息，绝不新增任何描述之外的内容，也不要逐帧流水账式复述。

另外输出 dropped：被你舍弃或模糊化的细节清单，每条在括号里附一句取舍理由，如 "墙上挂着三幅画（仅一段提及，数字类易幻觉）"。

只输出一个 JSON 对象，不要任何解释：
{{"long": "一段完整的长描述（不限字数）", "dropped": ["细节（取舍理由）", "..."]}}

{recaps}"""

FUSE_SHORT_KEYS = {"short"}
FUSE_SHORT_PROMPT = """下面是一段视频的完整中文描述。请把它提炼成**一条不超过 30 个字**的中文短字幕——这是最终训练用字幕。

要求：
1. 必须是一句自然通顺的话：核心主体+核心动作，放得下再加最有辨识度的一个外观或场景细节；不要罗列、不要分号顿号堆砌、不用"他/她/它/这"开头。
2. 只能使用描述里已有的信息，绝不新增任何细节；引号内的画面文字放不下就不写（长描述里已有）。
3. 超过 30 个字就舍弃次要细节，直到放得下为止。
4. gloss：短字幕里能在画面中被检测到的具体名词的英文（小写，1-4 个），如 ["man", "kitchen"]；抽象概念、动词不要放。

只输出一个 JSON 对象，不要任何解释：
{{"short": {{"text": "一条不超过30字的训练短字幕", "gloss": ["..."]}}}}

完整描述：
{long}"""

# --mode recap-slot (§5.6b, maintainer 2026-07-28): ASID-style FIXED-SLOT
# output — line format instead of JSON. Slots: 主体(<=2)/动作(<=2)/场景(1)/
# 文字(1); empty slot MUST be 「无」 (fixed slots invite fill-in hallucination
# -- the 无-discipline is the guardrail ASID leaves implicit). Line grammar is
# regex-parseable, so no extract_json, near-zero parse failure, and ~40-50%
# fewer decode tokens than the JSON caption-set. Prompt used RAW (no .format).
SLOT_RECAP_PROMPT = """你是一个严谨的视频画面描述助手。下面是从同一段视频中按时间先后均匀抽取的若干帧画面。
请按下面四个固定栏目描述画面，每行一条，格式为「栏目名：描述 [gloss]」。

栏目与行数：
主体：视频里的人/物及其外观特征（最多 2 行；有几个主要主体写几个）
动作：主体的动作及其先后（最多 2 行；先后用"先""随后""最后"表达）
场景：场景与背景环境（1 行）
文字：画面上清晰可辨的文字，用中文引号“”原样照抄（1 行）

严格遵守：
1. 每行描述不超过 28 个字，语法完整、可独立成句，不用"他/她/它/这"开头。
2. 只写画面中清晰可见的内容；看不清、被遮挡、需要推测的一律省略。某栏目在画面中不存在时，该行写「无」——绝不为了填满栏目而编造。
3. 不写画面之外的信息：不推断地点名称、人物身份、情绪动机、前因后果；禁止出现帧号、秒数、时间戳或"第几张图/第几帧/视频/画面"这类词。
4. 文字栏只在清晰可辨时原样照抄，不翻译、不补全、不猜测残缺文字；辨认不出就写「无」。
5. 数量只有很少且能数清时才给具体数字，否则用"几""一些""一群"等模糊量词。
6. 行尾方括号内给该行能在画面中被检测到的具体名词的英文（小写，逗号分隔，0-4 个）；没有可检测名词写 []；「无」行不加方括号。
7. 只输出这些行，不要任何解释、编号、标题或空行。

输出示例：
主体：一名穿红色围裙的中年男子 [man,apron]
主体：案板上放着一把菜刀 [knife]
动作：先把黄瓜切成薄片，随后装进盘中 []
场景：明亮的家用厨房操作台 [kitchen]
文字：无"""

SLOT_ATTRS = {"主体": "subject", "动作": "action", "场景": "scene",
              "文字": "text"}
SLOT_ATTRS_ZH = {v: k for k, v in SLOT_ATTRS.items()}
SLOT_CAPS = {"subject": 2, "action": 2, "scene": 1, "text": 1}
SLOT_LINE = re.compile(
    r"^\s*(主体|动作|场景|文字)\s*[:：]\s*(.+?)\s*(?:\[([^\[\]]*)\])?\s*$")


def parse_slot_lines(text):
    """(§5.6b) Parse fixed-slot line output -> [{text, gloss, attr}].
    「无」 slots are skipped, per-slot caps enforced, non-matching lines
    ignored (the model sometimes prefixes chatter -- lines without a slot
    name simply don't parse)."""
    out, counts = [], {}
    for line in (text or "").splitlines():
        m = SLOT_LINE.match(line)
        if not m:
            continue
        attr = SLOT_ATTRS[m.group(1)]
        body = m.group(2).strip()
        if not body or body.strip("。 ") == "无":
            continue
        if counts.get(attr, 0) >= SLOT_CAPS[attr]:
            continue
        gloss = [g.strip().lower() for g in (m.group(3) or "").split(",")
                 if g.strip()]
        out.append({"text": body, "gloss": gloss, "attr": attr})
        counts[attr] = counts.get(attr, 0) + 1
    return out


def slot_recap_cells(verdict):
    """(maintainer 2026-07-28) Rebuild per-pass ATTRIBUTE-LABELLED recap
    strings from an attribute-style verdict (recap-slot OR recap-struct):
    views carry {attr, src_recap}, while verdict.recaps strips the labels.
    Returns a list aligned with verdict.recaps ("【主体】... 【动作】..."),
    or None when the verdict has no attributed views (dense mode) -- caller
    falls back to verdict.recaps. Display-side only, so it works on cached
    jsonl with zero re-run. Caveat: views were deduped across passes at
    generation time, so a line repeated verbatim in a later pass shows only
    in its first pass; such passes fall back to the unlabelled recap text
    when they end up with no attributed line at all."""
    vd = verdict or {}
    recaps = [c if isinstance(c, str) else "" for c in (vd.get("recaps") or [])]
    by_pass = {}
    for v in vd.get("views") or []:
        if not (isinstance(v, dict) and isinstance(v.get("src_recap"), int)):
            continue
        zh = SLOT_ATTRS_ZH.get(v.get("attr"))
        t = v.get("text")
        if zh and isinstance(t, str) and t.strip():
            by_pass.setdefault(v["src_recap"], []).append(f"【{zh}】{t.strip()}")
    if not by_pass:
        return None
    return [" ".join(by_pass[i]) if by_pass.get(i) else recaps[i]
            for i in range(len(recaps))]


# --mode refine (§5.8-1, ASID-style selective refinement, arXiv 2602.13013):
# ONE minimal-edit repair call for views that FAILED Panel-V. Repair-then-
# reverify replaces delete-on-fail -- a failed check loses one phrase, not the
# whole view. Consumed via .format -> doubled braces.
REFINE_KEYS = {"views"}
REFINE_PROMPT = """下面每条中文短句是同一段视频的候选字幕，括号内标注了它未通过画面检验的原因（画面中检测不到某物体，或找不到某段文字）。请逐条做最小修改：删去或改写未通过检验的部分，其余用词尽量原样保留；改后每条仍不超过 30 个字、语法完整。若删去问题部分后信息所剩无几，该条输出空字符串 ""。

规则：
1. 只修问题处，绝不新增任何画面细节。
2. gloss：修改后短句里能在画面中被检测到的具体名词的英文（小写，0-4 个）。
3. 只输出一个 JSON 对象，条数与输入相同、顺序一一对应，不要任何解释：
{{"views": [{{"text": "修改后短句或空串", "gloss": ["..."]}}, ...]}}

待修短句：
{items}"""


def salvage_struct_items(text):
    """(§5.8-2a) Best-effort per-item rescue from malformed struct JSON: find
    each {...} object and keep the ones that parse with a non-empty text."""
    items = []
    for m in re.finditer(r"\{[^{}]*\}", text or ""):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and isinstance(o.get("text"), str) \
                and o["text"].strip():
            items.append(o)
    return items


# --mode recap-struct: OwlCap-style caption-SET output (arXiv 2508.18634's
# set idea, adopted offline -- §5.6): ONE decode emits 5-8 attribute-tagged
# short captions (subject/action/scene/text) with inline en_gloss, so the
# derive stage is subsumed (D8's gloss rides this call). The prompt is used
# RAW (no .format), so JSON braces below are single, not doubled. Passes stay
# independent (temperature + frame jitter) -- the cross-pass vote (§5.5)
# only means something over independent samples.
STRUCT_KEYS = {"captions"}
STRUCT_RECAP_PROMPT = """你是一个严谨的视频画面描述助手。下面是从同一段视频中按时间先后均匀抽取的若干帧画面。
请把画面内容拆成 5-8 条不同侧面的中文短描述，合起来尽量覆盖所有真实可见的信息，同时绝不编造。

严格遵守以下规则：
1. 每条不超过 28 个字，语法完整、可独立成句，不用"他/她/它/这"开头；每条标注一个属性 attr：
   subject（主体及外观特征）/ action（动作及先后）/ scene（场景与背景环境）/ text（画面文字）。
   同一属性可以出现多条（如多个主体、先后两个动作）；画面中不存在的属性完全不要输出。
2. 只描述画面中清晰可见的内容。看不清、不确定、被遮挡或需要推测的一律省略，不要猜测。
3. 不要描述画面之外的信息：不推断拍摄地点、时间、人物身份、情绪动机、前因后果。禁止出现
   帧号、秒数、时间戳或"第几张图/第几帧/视频/画面/图片"这类元描述词；动作的先后用
   "先""接着""随后""最后"表达。
4. 屏幕上的文字（字幕、标题、标牌、水印等）只有清晰可辨时才用中文引号“”原样照抄，
   不翻译、不补全、不猜测残缺文字；无法辨认就不输出 text 条目。
5. 数量只有在很少且能明确数清时才给具体数字，否则用"几""一些""一群"等模糊量词。
6. gloss：该条里能在画面中被检测到的具体名词的英文（小写，1-4 个），如 ["man", "kitchen"]；
   抽象概念、动词、顺序词不要放；没有可检测名词就给 []。
7. 只输出一个 JSON 对象，不要任何解释或多余文字，格式：
{"captions": [{"attr": "subject", "text": "……", "gloss": ["…"]}, {"attr": "action", "text": "……", "gloss": []}]}"""

# --mode derive: per-recap S7-style view derivation (master plan D5 A_llm).
# Text-only (the dense recap is the sole evidence; showing frames again would
# let the deriver inject NEW details, breaking the closed-set rule). gloss =
# detectable English noun phrases per view -- the GroundingDINO query handle
# (master plan D8: en_gloss rides the derivation call, not S1).
DERIVE_KEYS = {"views"}
DERIVE_PROMPT = """你会看到一段视频的中文稠密描述。请把它改写成 2-3 条互补的中文检索短句，并给每条短句配英文检测标签。

规则：
1. 每条短句不超过 28 个字，语法完整、可独立成句，不用"他/她/它/这"开头。
2. 2-3 条合起来覆盖原描述的主要信息，彼此侧重不同（如：主体+动作 / 场景环境 / 画面文字），不要互相复述。
3. 只能使用原描述里已有的信息，不得新增任何细节；原描述里用引号“”照抄的画面文字，短句中原样保留。
4. gloss：该短句里能在画面中被检测到的具体名词的英文（小写，1-4 个），如 ["man", "kitchen", "knife"]；抽象概念、动词、顺序词不要放。
5. 只输出一个 JSON 对象，不要任何解释或多余文字，格式：
{{"views": [{{"text": "短句一", "gloss": ["...", "..."]}}, {{"text": "短句二", "gloss": ["..."]}}]}}

原描述：
{dense}"""

# --mode condense: C_fuse recompose (master plan §5.5). Input = voted units
# from fuse_condense.py (cross-recap support >= k*, so every unit already
# carries multi-sample evidence); the LLM's only job is PACKING them densely
# into <=28-char views -- closed set, text-only, greedy. Support counts are
# shown so the model prioritizes high-vote units when trimming for length.
CONDENSE_KEYS = {"views"}
CONDENSE_PROMPT = """你会看到从同一段视频的多次独立描述中投票筛选出的若干条可靠信息单元（括号内为票数）。请把它们改写成 2-5 条互补的中文检索短句。

规则：
1. 每条短句不超过 28 个字，语法完整、可独立成句，不用"他/她/它/这"开头。
2. 把能自然合并的单元浓缩进同一条短句（如同一主体的外观+动作+位置），让每条短句信息密度尽量高；所有单元的信息合起来要全部覆盖，不要互相复述。宁可多写一条短句，也不要丢掉任何单元的信息；只有 2-5 条实在装不下时，才允许舍弃票数最低的单元。
3. 只能使用单元里已有的信息，不得新增任何细节；引号“”内的文字原样保留、不改一字。
4. gloss：该短句里能在画面中被检测到的具体名词的英文（小写，1-4 个），如 ["man", "kitchen"]；抽象概念、动词不要放。
5. 只输出一个 JSON 对象，不要任何解释：
{{"views": [{{"text": "短句一", "gloss": ["..."]}}, {{"text": "短句二", "gloss": ["..."]}}]}}

信息单元：
{units}"""


def resolve_video_path(video_root, rec, video):
    """First existing of:
      1. video_root/<video>
      2. video_root/<InternVId-... suffix of video>    (drops a host-specific
         prefix such as "videos/" that does not exist on another machine)
      3. video_root/<InternVId-... suffix of abs_path>
      4. raw abs_path
    else candidate 1, so the error message still shows what was tried."""
    cands = []
    if video:
        cands.append(os.path.join(video_root, str(video)))
        m = re.search(r"InternVId[^/]*/.*", str(video))
        if m:
            cands.append(os.path.join(video_root, m.group(0)))
    ap = rec.get("abs_path")
    if ap:
        m = re.search(r"InternVId[^/]*/.*", str(ap))
        if m:
            cands.append(os.path.join(video_root, m.group(0)))
        cands.append(str(ap))
    seen, uniq = set(), []
    for c in cands:                      # 2 and 3 collide when video==abs_path tail
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    for c in uniq:
        if os.path.exists(c):
            return c
    return uniq[0] if uniq else ""


_FRAME_BACKEND = None  # resolved once, reported so the choice is visible in logs


def _seg_centre_idx(n, num_frames):
    """TSN-style segment centers: InternVid clips are shot-cut, so the first/last
    frames are often transitions/black frames — never sample the boundaries."""
    return ((np.arange(num_frames) + 0.5) / num_frames * n).astype(int)


def _decode_decord(path, num_frames):
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    n = len(vr)
    if n == 0:
        raise RuntimeError("empty video")
    return vr.get_batch(_seg_centre_idx(n, num_frames)).asnumpy()


def _decode_opencv(path, num_frames):
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            raise RuntimeError("empty video")
        out = []
        for i in _seg_centre_idx(n, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                raise RuntimeError(f"read failed at frame {i}")
            out.append(fr[:, :, ::-1])          # BGR -> RGB
        return np.stack(out)
    finally:
        cap.release()


def _decode_pyav(path, num_frames):
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        n = stream.frames
        if n <= 0:                              # some containers don't store a count
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
            if not frames:
                raise RuntimeError("empty video")
            return np.stack([frames[i] for i in _seg_centre_idx(len(frames), num_frames)])
        wanted = set(int(i) for i in _seg_centre_idx(n, num_frames))
        got = {}
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                got[i] = frame.to_ndarray(format="rgb24")
                if len(got) == len(wanted):
                    break
        if not got:
            raise RuntimeError("empty video")
        order = [i for i in _seg_centre_idx(n, num_frames) if int(i) in got]
        return np.stack([got[int(i)] for i in order])


_BACKENDS = {"decord": _decode_decord, "opencv": _decode_opencv, "pyav": _decode_pyav}


def _resolve_backend(preference):
    """decord is preferred so the judge samples exactly what vlm_audit_hf.py fed
    the main model. opencv/pyav exist because a relay host may have no route to
    PyPI to install decord; all three use the same index formula, but a different
    decoder can report a different frame count for the same file, so the chosen
    backend is logged and worth recording next to the agreement numbers."""
    global _FRAME_BACKEND
    if _FRAME_BACKEND:
        return _FRAME_BACKEND
    order = [preference] if preference != "auto" else ["decord", "opencv", "pyav"]
    for name in order:
        try:
            __import__({"decord": "decord", "opencv": "cv2", "pyav": "av"}[name])
        except ImportError:
            continue
        _FRAME_BACKEND = name
        print(f"[frames] backend={name}", file=sys.stderr)
        return name
    raise RuntimeError(
        "no video backend available — install one of: decord, opencv-python, av")


def sample_frames_b64(path, num_frames, max_side, quality=80, backend="auto"):
    frames = _BACKENDS[_resolve_backend(backend)](path, num_frames)
    urls = []
    for fr in frames:
        img = Image.fromarray(fr)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        urls.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return urls


def extract_json(text, required=None):
    """First JSON object in `text` containing all of `required` (default: a full
    verdict). Parameterised so the text-only translation pass can demand its own
    key set instead of the combined-rubric one."""
    required = VERDICT_KEYS if required is None else required
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if required <= set(obj.keys()):
                        return obj
                except json.JSONDecodeError:
                    pass
    raise ValueError("no valid verdict JSON in response")


def audit_one(rec, args, session):
    video = rec["video"]
    out = {"video": video, "stratum": rec.get("stratum"),
           "s1": rec.get("s1"), "s0_flags": rec.get("s0_flags")}
    path = resolve_video_path(args.video_root, rec, video)
    try:
        frames = sample_frames_b64(path, args.frames, args.max_side,
                                   backend=args.frame_backend)
    except Exception as ex:  # noqa: BLE001 - decode failure is itself a verdict
        out["verdict"] = {"video_readable": False, "action": "drop",
                          "error": f"decode: {type(ex).__name__}"}
        return out
    prompt = PROMPT.format(n_frames=args.frames,
                           en=(rec.get("en") or "(无)"), zh=(rec.get("zh") or "(空)"))
    content = [{"type": "image_url", "image_url": {"url": u}} for u in frames]
    content.append({"type": "text", "text": prompt})
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    if not args.allow_thinking:
        # Qwen3.x ships a thinking mode that is ON by default. Left on, the model
        # spends the whole max_tokens budget inside <think> and gets truncated
        # before it ever emits the verdict JSON -- and a truncated block has no
        # closing </think>, so extract_json's strip cannot even recover it. Every
        # record would come back {"error": "no valid verdict JSON in response"}.
        # Matches the in-house client for these endpoints.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    last_err = None
    for attempt in range(args.retries + 1):
        try:
            r = session.post(f"{args.api_base}/chat/completions", json=body,
                             timeout=args.timeout,
                             headers={"Authorization": f"Bearer {args.api_key}"})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            out["verdict"] = extract_json(text)
            return out
        except Exception as ex:  # noqa: BLE001
            last_err = f"{type(ex).__name__}: {ex}"
            if attempt < args.retries:
                # exponential backoff: a shared endpoint under load returns 429/503,
                # and retrying instantly just adds to the pile
                time.sleep(2 ** attempt)
    out["verdict"] = {"error": last_err}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="sample jsonl from make_audit_sample.py")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--frame-backend", default="auto",
                    choices=["auto", "decord", "opencv", "pyav"],
                    help="video decoder; auto prefers decord (what the main "
                         "model used), then opencv, then pyav")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print [progress] every N verdicts (was hardcoded 200, "
                         "which looks like a stall on a 500-record run)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-request timeout; 600 matches the in-house client for "
                         "these endpoints (a shared 27B with 8 frames can be slow)")
    ap.add_argument("--allow-thinking", action="store_true",
                    help="do NOT send chat_template_kwargs.enable_thinking=False. "
                         "Only for endpoints that reject that field; on Qwen3.x "
                         "leaving thinking on makes every verdict fail to parse")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.output):  # resume
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[resume] {len(done)} verdicts already present", file=sys.stderr)

    todo = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["video"] not in done:
                todo.append(rec)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[audit] {len(todo)} clips to judge with {args.model}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    lock = threading.Lock()
    n_done = 0
    with open(args.output, "a", encoding="utf-8") as fout, \
         requests.Session() as session, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # as_completed, NOT pool.map: map yields strictly in submission order, so a
        # single slow clip blocks every finished result behind it from being written
        # -- with --timeout 600 and retries that is a ~30 min stall during which
        # both the output file and the progress line look frozen.
        futures = [pool.submit(audit_one, r, args, session) for r in todo]
        for fut in as_completed(futures):
            out = fut.result()
            with lock:
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % args.progress_every == 0:
                    print(f"[progress] {n_done}/{len(todo)}", file=sys.stderr, flush=True)
    print(f"[done] wrote {n_done} verdicts to {args.output}")


if __name__ == "__main__":
    main()
