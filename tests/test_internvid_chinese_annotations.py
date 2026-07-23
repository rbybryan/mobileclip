import json
import tempfile
from pathlib import Path

from preprocess.prepare_internvid_chinese_annotations import convert_annotations


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_merge_reviews_and_split_shards():
    source_records = [
        {
            "video": "InternVId-FLT_1/a.mp4",
            "video_key": "a.mp4",
            "raw_caption_zh_translated": "原始译文一",
        },
        {
            "video": "InternVId-FLT_2/b.mp4",
            "video_key": "b.mp4",
            "raw_caption_zh_translated": "原始译文二",
        },
        {
            "video": "InternVId-FLT_2/c.mp4",
            "video_key": "c.mp4",
            "raw_caption_zh_translated": "不确定译文",
        },
    ]
    manual_records = [
        {
            "_qa_source_line_no": 2,
            "video": "InternVId-FLT_2/b.mp4",
            "video_key": "b.mp4",
            "_human_review": {
                "source_line_no": 2,
                "verdict": "修改后通过",
                "corrected_translation": "人工修订译文",
            },
        },
        {
            "_qa_source_line_no": 3,
            "video": "InternVId-FLT_2/c.mp4",
            "video_key": "c.mp4",
            "_human_review": {
                "source_line_no": 3,
                "verdict": "不确定",
                "corrected_translation": "仍不确定",
            },
        },
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / "source.jsonl"
        manual_path = temp_path / "manual.jsonl"
        output_dir = temp_path / "output"
        write_jsonl(source_path, source_records)
        write_jsonl(manual_path, manual_records)

        summary = convert_annotations(
            source_path,
            manual_path,
            output_dir,
            expected_input_count=3,
            progress_every=0,
        )

        assert summary["input_count"] == 3
        assert summary["output_count"] == 2
        assert summary["action_counts"] == {
            "manual_override": 1,
            "translated": 1,
            "uncertain_dropped": 1,
        }
        assert json.load((output_dir / "InternVId-FLT_1.json").open()) == [
            {"video": "InternVId-FLT_1/a.mp4", "caption": "原始译文一"}
        ]
        assert json.load((output_dir / "InternVId-FLT_2.json").open()) == [
            {"video": "InternVId-FLT_2/b.mp4", "caption": "人工修订译文"}
        ]
        assert json.load((output_dir / "InternVId-FLT_11.json").open()) == []


if __name__ == "__main__":
    test_merge_reviews_and_split_shards()
    print("InternVid Chinese annotation conversion test passed")
