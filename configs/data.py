import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

# ---------------------------------------------------------------------------
# Dataset locations.
#
# All annotation/video paths below are resolved against two roots so that a
# machine with datasets stored elsewhere only needs to set environment
# variables (no code edits):
#
#   MVC_DATA_DIR  root of the raw video folders   (default: "data")
#   MVC_ANNO_DIR  root of the annotation jsons    (default: "anno_downstream")
#
# e.g.  export MVC_DATA_DIR=/mnt/datasets/mobileviclip/data
#       export MVC_ANNO_DIR=/mnt/datasets/mobileviclip/anno_downstream
#
# Names prefixed with "__" are intentionally hidden from the config loader
# (utils/config.py drops names starting with "__").
# ---------------------------------------------------------------------------
__data_dir = __os.environ.get("MVC_DATA_DIR", "data")
__anno_dir = __os.environ.get("MVC_ANNO_DIR", "anno_downstream")


def __anno(rel_path):
    return __os.path.join(__anno_dir, rel_path)


def __data(rel_path):
    return __os.path.join(__data_dir, rel_path)


# ============== pretraining datasets=================
available_corpus = dict()

for __i in range(1, 12):
    available_corpus[f"internvid_v{__i}"] = dict(
        anno_path=__anno(f"InternVid/InternVid-10M-flt_{__i}.json"),
        data_root=__data(f"video_sample/InternVId-FLT_{__i}"),
        media_type="video",
        jump_filter=True,
    )

available_corpus["internvid_all"] = [
    available_corpus[f"internvid_v{__i}"] for __i in range(1, 12)
]

# ============== for validation =================
available_corpus["msrvtt_1k_test"] = dict(
    anno_path=__anno("msrvtt_1k_test.json"),
    data_root=__data("MSRVTT_Videos"),
    media_type="video"
)

available_corpus["didemo_ret_test"] = dict(
    anno_path=__anno("didemo_ret_test.json"),
    data_root=__data("didemo/processed_videos/test"),
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["anet_ret_val"] = dict(
    anno_path=__anno("anet_ret_val.json"),
    data_root=__data("anet_1.3_video_val_resize"),
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["k400_act_val"] = dict(
    anno_path=__anno("kinetics400_validate.json"),
    data_root=__data("kinetics_400_val_10s_320p"),
    media_type="video",
    prompt="kinetics",
    is_act_rec=True,
)

available_corpus["ucf101_act_test"] = dict(
    anno_path=__anno("ucf101_act_test.json"),
    data_root=__data("ucf101/videos"),
    media_type="video",
    prompt="kinetics",
    is_act_rec=True,
)


available_corpus["hmdb51_act_test"] = dict(
    anno_path=__anno("hmdb51_act_test.json"),
    data_root=__data("hmdb51/videos_test"),
    media_type="video",
    prompt="kinetics",
    is_act_rec=True,
)

available_corpus["ssv2_mc_val"] = dict(
    anno_path=__anno("ssv2_mc_val.json"),
    data_root=__data("ssv2/20bn-something-something-v2"),
    media_type="video",
)
