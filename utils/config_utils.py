import logging
import os
import sys
import json
import torch.distributed as dist
from os.path import dirname, join

from utils.config import Config
from utils.distributed import init_distributed_mode, is_main_process
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def setup_config():
    """Conbine yaml config and command line config with OmegaConf.
    Also converts types, e.g., `'None'` (str) --> `None` (None)
    """
    config = Config.get_config()
    if config.debug:
        config.wandb.enable = False
    return config


def setup_evaluate_config(config):
    """setup evaluation default settings, e.g., disable wandb"""
    assert config.evaluate
    config.wandb.enable = False
    if config.output_dir is None:
        config.output_dir = join(dirname(config.pretrained_path), "eval")
    return config


def setup_output_dir(output_dir, excludes=["code"]):
    """ensure not overwritting an exisiting/non-empty output dir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    else:
        existing_dirs_files = os.listdir(output_dir)  # list
        remaining = set(existing_dirs_files) - set(excludes)
        remaining = [e for e in remaining if "slurm" not in e]
        remaining = [e for e in remaining if ".out" not in e]
        # assert len(remaining) == 0, f"remaining dirs or files: {remaining}"
        logger.warn(f"remaining dirs or files: {remaining}")


def setup_deepspeed_zero_config(stage):
    # We currently set ZeRO based on stage:
    if stage == 1:
        return {"stage": 1, "reduce_bucket_size": 5e8}
    if stage == 2:
        return {
            "stage": 2,
            "contiguous_gradients": False,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "offload_optimizer": {
                "device": "cpu"
            },
        }
    if stage == 3:
        return {
            "stage": 3,
            "contiguous_gradients": True,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_prefetch_bucket_size": 1e7,
            "stage3_param_persistence_threshold": 1e5,
            "reduce_bucket_size": 1e7,
            "sub_group_size": 1e9,
            "offload_optimizer": {
                "device": "cpu"
            },
            "offload_param": {
                "device": "cpu"
            }
        }
    
    raise ValueError("Wrong stage for deepspeed {}".format(stage.stage))

def setup_deepspeed_config(config):
    config.deepspeed_config = os.path.join(config.output_dir, "deepspeed_config.json")
    opts = config.optimizer
    logger.info(f'Write deepspeed config to {config.deepspeed_config}')
    if not is_main_process():
        return config
    
    os.makedirs(config.output_dir, exist_ok=True)

    # DeepSpeed 默认用 FusedAdam,而 FusedAdam 要在首次运行时用 nvcc **现编** CUDA
    # 扩展。系统 nvcc 比系统 g++ 老时(101vector 上 nvcc 解析 gcc 11 的
    # std_function.h 直接报 "parameter packs not expanded",见 TROUBLESHOOTING.md)
    # 这一步必然失败,训练起不来。torch_adam 让 DeepSpeed 改用 torch.optim.AdamW ——
    # 同一个算法,只是没有融合核,不需要任何编译。要融合核时:`deepspeed.torch_adam False`。
    # 注:adam_w_mode / torch_adam 必须放在 params 里,DeepSpeed 从那里 pop,
    # 放在 optimizer 顶层会被静默忽略。
    torch_adam = config.deepspeed.get("torch_adam", True)
    adam_params = {
        "lr": opts.lr,
        "weight_decay": opts.weight_decay,
        "betas": [
            opts.opt_betas[0],
            opts.opt_betas[1],
        ],
        "eps": 1e-8,
        "adam_w_mode": True,
    }
    if torch_adam:
        # torch.optim.AdamW 不接受 bias_correction(它本来就一直做偏置校正)
        adam_params["torch_adam"] = True
    else:
        adam_params["bias_correction"] = True

    with open(config.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": config.batch_size * dist.get_world_size(),
            "train_micro_batch_size_per_gpu": config.batch_size,
            "steps_per_print": 100,
            "optimizer": {
                "type": "Adam",
                "params": adam_params,
            }
        }
        if config.deepspeed.stage != 0:
            ds_config["zero_optimization"] = setup_deepspeed_zero_config(config.deepspeed.stage)

        if config.use_half_precision:
            if config.get('use_bf16', False):
                ds_config["bf16"] = {
                "enabled": True
                }
            else:
                ds_config["fp16"] = {
                    "enabled": True,
                    "auto_cast": False,
                    "loss_scale": 0,
                    # 可从配置/命令行覆盖:`deepspeed.initial_scale_power 5`。
                    # 默认见 scripts/pretraining/clip/mobileviclip_small/config.py
                    # 里的说明(与可学习 temp 的梯度量级有关)。
                    "initial_scale_power": config.deepspeed.get("initial_scale_power", 16),
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "consecutive_hysteresis": False,
                    "min_loss_scale": 1
                } 
        else:
            assert config.deepspeed.stage == 0, "You must use fp16 or bf16 when using ZERO!!!"
            
        if opts.get("max_grad_norm", -1) > 0:
            ds_config["gradient_clipping"] = opts.max_grad_norm

        writer.write(json.dumps(ds_config, indent=2))
    
    return config


def setup_main():
    """
    Setup config, logger, output_dir, etc.
    Shared for pretrain and all downstream tasks.
    """
    # try:
    config = setup_config()
    if hasattr(config, "evaluate") and config.evaluate:
        config = setup_evaluate_config(config)
    init_distributed_mode(config)

    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        config = setup_deepspeed_config(config)
    
    if is_main_process():
        setup_output_dir(config.output_dir, excludes=["code"])
        setup_logger(output=config.output_dir, color=True, name="vindlu")
        logger.info(f"config: {Config.pretty_text(config)}")
        Config.dump(config, os.path.join(config.output_dir, "config.json"))
    
    dist.barrier()

    return config
