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
