#!/usr/bin/env python3
"""Merge translated InternVid JSONL records with manual reviews.

The output is eleven minimal JSON arrays that can be consumed by the
MobileViCLIP pretraining dataset without retaining the large audit payload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


SHARD_IDS = tuple(range(1, 12))
SHARD_RE = re.compile(r"^InternVId-FLT_(1[01]|[1-9])/")
ACCEPTED_VERDICTS = {"通过", "修改后通过"}
DROP_VERDICT = "拒绝/删除"
UNCERTAIN_VERDICT = "不确定"


def read_json_object(raw_line: str, path: Path, line_no: int) -> dict[str, Any]:
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object in {path} at line {line_no}")
    return record


def load_manual_reviews(path: Path) -> dict[int, dict[str, Any]]:
    reviews: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for file_line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = read_json_object(raw_line, path, file_line_no)
            review = record.get("_human_review")
            if not isinstance(review, dict):
                raise ValueError(
                    f"Missing _human_review object in {path} at line {file_line_no}"
                )
            source_line_no = review.get("source_line_no")
            if not isinstance(source_line_no, int) or source_line_no < 1:
                raise ValueError(
                    f"Invalid source_line_no in {path} at line {file_line_no}"
                )
            if record.get("_qa_source_line_no") != source_line_no:
                raise ValueError(
                    f"Source line mismatch in {path} at line {file_line_no}"
                )
            if source_line_no in reviews:
                raise ValueError(f"Duplicate review for source line {source_line_no}")
            reviews[source_line_no] = {
                "verdict": review.get("verdict"),
                "corrected_translation": review.get("corrected_translation"),
                "video": record.get("video"),
                "video_key": record.get("video_key"),
            }
    return reviews


def select_caption(
    record: dict[str, Any],
    review: dict[str, Any] | None,
    uncertain_policy: str,
) -> tuple[str | None, str]:
    raw_caption = record.get("raw_caption_zh_translated")
    if not isinstance(raw_caption, str) or not raw_caption.strip():
        raise ValueError("raw_caption_zh_translated must be a non-empty string")
    raw_caption = raw_caption.strip()

    if review is None:
        return raw_caption, "translated"

    verdict = review["verdict"]
    corrected = review.get("corrected_translation")
    corrected = corrected.strip() if isinstance(corrected, str) else ""
    if verdict in ACCEPTED_VERDICTS:
        if not corrected:
            raise ValueError(f"Review verdict {verdict!r} has an empty correction")
        return corrected, "manual_override"
    if verdict == DROP_VERDICT:
        return None, "rejected"
    if verdict == UNCERTAIN_VERDICT:
        if uncertain_policy == "drop":
            return None, "uncertain_dropped"
        if uncertain_policy == "corrected":
            return corrected or raw_caption, "uncertain_corrected"
        return raw_caption, "uncertain_raw"
    raise ValueError(f"Unsupported manual-review verdict: {verdict!r}")


def shard_id_from_video(video: str) -> int:
    match = SHARD_RE.match(video)
    if match is None:
        raise ValueError(
            "video must be relative and start with InternVId-FLT_1 through "
            f"InternVId-FLT_11, got {video!r}"
        )
    return int(match.group(1))


def convert_annotations(
    source_path: Path,
    manual_path: Path,
    output_dir: Path,
    *,
    uncertain_policy: str = "drop",
    expected_input_count: int | None = None,
    overwrite: bool = False,
    progress_every: int = 100_000,
) -> dict[str, Any]:
    if uncertain_policy not in {"drop", "raw", "corrected"}:
        raise ValueError(f"Unsupported uncertain policy: {uncertain_policy}")

    reviews = load_manual_reviews(manual_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        shard_id: output_dir / f"InternVId-FLT_{shard_id}.json"
        for shard_id in SHARD_IDS
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Output files already exist: {names}")

    temp_paths = {
        shard_id: output_dir / f".{path.name}.tmp-{os.getpid()}"
        for shard_id, path in final_paths.items()
    }
    handles: dict[int, Any] = {}
    shard_counts: Counter[int] = Counter()
    action_counts: Counter[str] = Counter()
    seen_reviews: set[int] = set()
    input_count = 0
    started = time.monotonic()

    try:
        for shard_id, temp_path in temp_paths.items():
            handles[shard_id] = temp_path.open("x", encoding="utf-8")
            handles[shard_id].write("[\n")

        with source_path.open("r", encoding="utf-8") as source:
            for input_count, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    raise ValueError(
                        f"Blank line in {source_path} at line {input_count}"
                    )
                record = read_json_object(raw_line, source_path, input_count)
                video = record.get("video")
                if not isinstance(video, str) or not video:
                    raise ValueError(f"Missing video at source line {input_count}")
                shard_id = shard_id_from_video(video)

                review = reviews.get(input_count)
                if review is not None:
                    seen_reviews.add(input_count)
                    if review.get("video") != video:
                        raise ValueError(
                            f"Manual review video mismatch at source line {input_count}"
                        )
                    if review.get("video_key") != record.get("video_key"):
                        raise ValueError(
                            f"Manual review video_key mismatch at source line {input_count}"
                        )

                try:
                    caption, action = select_caption(
                        record, review, uncertain_policy=uncertain_policy
                    )
                except ValueError as exc:
                    raise ValueError(f"Source line {input_count}: {exc}") from exc
                action_counts[action] += 1
                if caption is not None:
                    output_record = {"video": video, "caption": caption}
                    handle = handles[shard_id]
                    if shard_counts[shard_id]:
                        handle.write(",\n")
                    handle.write(
                        json.dumps(
                            output_record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    shard_counts[shard_id] += 1

                if progress_every and input_count % progress_every == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"Processed {input_count:,} rows "
                        f"({input_count / elapsed:,.0f} rows/s)",
                        file=sys.stderr,
                        flush=True,
                    )

        if expected_input_count is not None and input_count != expected_input_count:
            raise ValueError(
                f"Expected {expected_input_count:,} input rows, got {input_count:,}"
            )
        unseen_reviews = sorted(set(reviews) - seen_reviews)
        if unseen_reviews:
            preview = ", ".join(str(value) for value in unseen_reviews[:10])
            raise ValueError(f"Manual reviews not found in source JSONL: {preview}")

        for shard_id in SHARD_IDS:
            handles[shard_id].write("\n]\n")
            handles[shard_id].close()
        handles.clear()
        for shard_id in SHARD_IDS:
            os.replace(temp_paths[shard_id], final_paths[shard_id])
    except BaseException:
        for handle in handles.values():
            handle.close()
        for temp_path in temp_paths.values():
            temp_path.unlink(missing_ok=True)
        raise

    return {
        "input_count": input_count,
        "output_count": sum(shard_counts.values()),
        "manual_review_count": len(reviews),
        "uncertain_policy": uncertain_policy,
        "action_counts": dict(sorted(action_counts.items())),
        "shard_counts": {
            f"InternVId-FLT_{shard_id}": shard_counts[shard_id]
            for shard_id in SHARD_IDS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MobileViCLIP Chinese InternVid JSON annotation shards."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manual-reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--uncertain-policy",
        choices=("drop", "raw", "corrected"),
        default="drop",
    )
    parser.add_argument("--expected-input-count", type=int)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_annotations(
        source_path=args.source,
        manual_path=args.manual_reviews,
        output_dir=args.output_dir,
        uncertain_policy=args.uncertain_policy,
        expected_input_count=args.expected_input_count,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
