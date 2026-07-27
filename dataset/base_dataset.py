import logging
import os
import random
try:
    from petrel_client.client import Client
except:
    Client = None
from torch.utils.data import Dataset
from .utils import load_image_from_path
from .av_utils import lazy_load_s3video

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# GATE-0 时间维容量审计:评测期的帧扰动开关(默认关闭 = 逐位等价基线)
#
# 只由环境变量 MVC_FRAME_PERTURB 打开,取值:
#   shuffle  随机置换 T 帧(按 index 确定性设种,可复现)
#   reverse  时间反转
#   freeze   全部替换成中间帧(单帧上界探针)
# 未设置 / 设为 none 时函数**原样返回同一个对象**,不做任何算子。
#
# ⚠️ 只在审计评测时设置,**绝不要在训练时打开**。
# 背景与判读见 notes/neural-inspired-round2.md §2。
# --------------------------------------------------------------------------
_PERTURB_WARNED = False


def maybe_perturb_frames(frames, index=0):
    """帧扰动(默认 no-op)。frames: [T, C, H, W]。"""
    mode = os.environ.get("MVC_FRAME_PERTURB", "").strip().lower()
    if not mode or mode == "none":
        return frames

    global _PERTURB_WARNED
    if not _PERTURB_WARNED:
        logger.warning(
            "MVC_FRAME_PERTURB=%s —— 帧顺序正在被人为扰动(GATE-0 审计模式)。"
            "训练时绝不应出现这一行。", mode
        )
        _PERTURB_WARNED = True

    import torch

    T = frames.shape[0]
    if mode == "reverse":
        return torch.flip(frames, dims=[0])
    if mode == "freeze":
        return frames[T // 2].unsqueeze(0).repeat(
            T, *([1] * (frames.dim() - 1))
        )
    if mode == "shuffle":
        # 按样本 index 确定性设种,保证复跑得到同一置换
        g = torch.Generator().manual_seed(
            int(os.environ.get("MVC_FRAME_PERTURB_SEED", 0)) * 1000003 + int(index)
        )
        return frames[torch.randperm(T, generator=g)]
    raise ValueError(
        f"未知的 MVC_FRAME_PERTURB={mode!r};可选 shuffle / reverse / freeze / none"
    )


# --------------------------------------------------------------------------
# 选帧基线(情形 B):从已解码的 T 帧母序列里挑 K 帧,模型真正只跑 K 帧
# ⇒ 与 num_frames_test=K 成本一致,可同台比较。默认关闭 = 逐位等价基线。
#
#   MVC_FRAME_SELECT    uniform | divmax | random | none(未设=none)
#   MVC_FRAME_SELECT_K  选几帧(必填;>= T 时不切片)
#   MVC_FRAME_SELECT_SEED  random 模式的种子(默认 0)
#
# 动机、成本诚实性与判读见 utils/frame_select.py 顶部与
# experiments/frame_select_results.md。⚠️ 只在评测时设置,不要在训练时打开。
# --------------------------------------------------------------------------
_SELECT_WARNED = False


def maybe_select_frames(frames, index=0):
    """选帧(默认 no-op)。frames: [T, C, H, W] -> [K, C, H, W]。"""
    mode = os.environ.get("MVC_FRAME_SELECT", "").strip().lower()
    if not mode or mode == "none":
        return frames

    raw_k = os.environ.get("MVC_FRAME_SELECT_K", "").strip()
    if not raw_k:
        raise ValueError("设了 MVC_FRAME_SELECT 就必须同时设 MVC_FRAME_SELECT_K")

    from utils.frame_select import select_indices

    idx = select_indices(
        mode,
        total=frames.shape[0],
        k=int(raw_k),
        frames=frames,
        index=index,
        seed=int(os.environ.get("MVC_FRAME_SELECT_SEED", 0)),
    )

    global _SELECT_WARNED
    if not _SELECT_WARNED:
        logger.warning(
            "MVC_FRAME_SELECT=%s K=%s —— 正在从 %d 帧里选帧(选帧基线实验)。"
            "训练时绝不应出现这一行。%s",
            mode, raw_k, frames.shape[0],
            "" if idx is not None else " 本次 K>=T,不切片。",
        )
        _SELECT_WARNED = True

    return frames if idx is None else frames[idx]


class BaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["audio", "image", "video", "audio_video"]
        self.data_root = None
        self.data_root_prefix = ""
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.audio_reader_type = None
        self.audio_sample_rate = None
        self.max_audio_length = None
        self.video_reader = None
        self.num_tries = None
        self.client = Client('~/petreloss.conf') if Client is not None else None
        self.trimmed30 = False

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index): # NOTE used for most ret_dataset
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            if self.media_type == "audio":
                anno["audio"] = self.data_root_prefix + os.path.join(self.data_root, anno["audio"])
            else:
                anno["image"] = self.data_root_prefix + os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path):
        try:
            if self.media_type == "image":
                return self.load_and_transform_media_data_image(index, data_path)
            elif self.media_type == "audio":
                return self.load_and_transform_media_data_audio(index, data_path)
            elif self.media_type == "video":
                return self.load_and_transform_media_data_video(index, data_path)
            elif self.media_type == "audio_video":
                return self.load_and_transform_media_data_audio_video(index, data_path)
            else:
                raise NotImplementedError(self.media_type)
        except Exception as e:
            logger.info(f"Something wrong when read {data_path}")
            raise e

    def load_and_transform_media_data_image(self, index, data_path):
        if type(data_path) is dict:
            image = load_image_from_path(data_path["image"], client=self.client)
            if "crop_bbox" in data_path.keys():
                bbox = data_path["crop_bbox"]
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                image = image[:, :, y0:y1, x0:x1]
                
            image = self.transform(image)
        else:

            image = load_image_from_path(data_path, client=self.client)
            image = self.transform(image)
        return image, index
    
    def load_and_transform_media_data_video(self, index, data_path):
        if type(data_path) is dict:
            if data_path['read_clip_from_video']:
                if self.trimmed30:
                    raise NotImplementedError("lazy_load_s3video does not support trimmed30")
                frames = lazy_load_s3video(data_path['video'], self.num_frames, data_path['video_start_frame'], data_path['video_end_frame'], self.client)
            else:
                raise NotImplementedError(data_path)
        else:
            max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
            frames, frame_indices, video_duration = self.video_reader(
                data_path, self.num_frames, self.sample_type, 
                max_num_frames=max_num_frames, client=self.client,
                trimmed30=self.trimmed30
            )

        # NOTE shared aug for video frames
        frames = self.transform(frames)
        # GATE-0 审计开关;未设 MVC_FRAME_PERTURB 时是纯 no-op(返回同一对象)
        frames = maybe_perturb_frames(frames, index)
        # 选帧基线开关;未设 MVC_FRAME_SELECT 时是纯 no-op(返回同一对象)
        # 放在扰动之后:扰动是诊断、选帧是候选方案,两者正交且不应同时开
        frames = maybe_select_frames(frames, index)
        return frames, index

