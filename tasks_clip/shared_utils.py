import copy
import logging
import os
import os.path as osp
from os.path import join

import torch
try:
    import deepspeed
except Exception as _e:  # not installed, or version conflicts (e.g. old
    # deepspeed + pydantic>=2 raises PydanticUserError at import time)
    deepspeed = None
    print(f"deepspeed unavailable ({type(_e).__name__}: {_e}) -- "
          "only configs with deepspeed.enable=False can run")
from torch.utils.data import ConcatDataset, DataLoader

from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

from models.backbones.internvideo2.mobileclip.modules.common.mobileone import reparameterize_model

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, pretrain=False, find_unused_parameters=False, num_steps_per_epoch=-1,reparameterize=False,
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    model = model_cls(config=config, is_pretrain=pretrain)

    model = model.to(torch.device(config.device))
    if config.use_half_precision:
        # `use_bf16` is the one precision key (same as config_utils.py and
        # pretrain.py); legacy `bf16` still honoured as a fallback for old
        # configs/commands that only set that spelling.
        if config.get('use_bf16', config.get('bf16', True)):
            logger.info("Change to bfloat16 for model")
            model = model.to(torch.bfloat16)
        else:
            logger.info("Change to float16 for model")
            model = model.half()
    tokenizer = model.tokenizer 
    

    model_without_ddp = model

    if hasattr(config, "deepspeed") and config.deepspeed.enable: 
        optimizer_params = create_optimizer(config.optimizer, model, return_group=True)
        scheduler = None
        scaler = None
    else:
        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.gpu],
                find_unused_parameters=find_unused_parameters,  # `False` for image-only task
            )

        optimizer = create_optimizer(config.optimizer, model)
        scheduler = create_scheduler(config.scheduler, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_half_precision) # This is never used actually if we fixed bf16

    start_epoch = 0
    global_step = 0

    # auto resume the latest checkpoint
    if config.get("auto_resume", False): 
        logger.info("Auto resuming")
        model_latest = join(config.output_dir, "ckpt_latest.pth")
        model_best = join(config.output_dir, "ckpt_best.pth")

        large_step_num = -1
        large_num = -1
        for p in os.listdir(config.output_dir):
            if 'ckpt_iter' in p:
                num = p.split('_iter')[1].split('.')[0]
                if str.isnumeric(num):
                    if int(num) > large_step_num:
                        large_step_num = int(num)
            elif 'ckpt_' in p:
                num = p.split('_')[1].split('.')[0]
                if str.isnumeric(num):
                    if int(num) > large_num:
                        large_num = int(num)
        if large_step_num != -1:
            logger.info(f"Load the latest step: {large_step_num}")
            model_latest = join(config.output_dir, f"ckpt_iter{large_step_num:02d}.pth")
        if large_num != -1 and (large_num + 1) * num_steps_per_epoch > large_step_num:
            logger.info(f"Load the latest epoch: {large_num}")
            model_latest = join(config.output_dir, f"ckpt_{large_num:02d}.pth")

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            if osp.isdir(model_latest):
                config.pretrained_path = model_latest
                config.resume = True
            elif osp.isdir(model_best):
                config.pretrained_path = model_best
                config.resume = True
            else:
                logger.info(f"Not found checkpoint in {config.output_dir}")
        else:
            if osp.isfile(model_latest):
                config.pretrained_path = model_latest
                config.resume = True
            elif osp.isfile(model_best):
                config.pretrained_path = model_best
                config.resume = True
            else:
                logger.info(f"Not found checkpoint in {config.output_dir}")

    # load pretrained model
    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        assert deepspeed is not None, (
            "config.deepspeed.enable=True but deepspeed failed to import "
            "(see message at startup). Either fix the env (e.g. "
            "pip install 'pydantic<2', or upgrade deepspeed) or run with "
            "the CLI override: `deepspeed.enable False`."
        )
        logger.info('Use deepspeed to initialize model!!!')
        model = model_without_ddp
        model, optimizer, _, _ = deepspeed.initialize(
            args=config, model=model, model_parameters=optimizer_params, dist_init_required=not config.distributed,
            lr_scheduler=lambda opt: create_scheduler(config.scheduler, opt)
        )
        if osp.isdir(config.pretrained_path):
            logger.info(f"Load pretrained model from {config.pretrained_path}")
            output_dir, tag = os.path.split(config.pretrained_path)
            if config.resume:
                _, client_state = model.load_checkpoint(output_dir, tag=tag, load_module_strict=False)
                global_step = model.global_steps
                assert num_steps_per_epoch > 0, "Please provide num_steps_per_epoch"
                start_epoch = global_step // num_steps_per_epoch
            else:
                _, client_state = model.load_checkpoint(
                    output_dir, tag=tag, load_module_strict=False, 
                    load_optimizer_states=False, load_lr_scheduler_states=False,
                    load_module_only=True
                )
    else:
        # DeepSpeed 存的是**目录式** ckpt(`ckpt_02.pth/mp_rank_00_model_states.pt`),
        # 而这一支(deepspeed.enable=False,评测配置的默认)只判 `osp.isfile` ⇒ 目录恒为 False
        # ⇒ **静默跳过加载**,最后只警告一句 "training from scratch",评测出来的是
        # 未训练的模型。2026-07-29 补评 sub1m_r0_s42 就这样出了一组看似正常、实为
        # MobileCLIP-S2 初始化 + 随机时序算子的数字。这里自动把目录解析到里面的权重文件。
        if osp.isdir(config.pretrained_path):
            _ds_inner = join(config.pretrained_path, "mp_rank_00_model_states.pt")
            if osp.isfile(_ds_inner):
                logger.info(
                    f"pretrained_path 是 DeepSpeed 目录式 ckpt,自动解析到 {_ds_inner}"
                )
                config.pretrained_path = _ds_inner
            else:
                raise FileNotFoundError(
                    f"pretrained_path 是目录但里面没有 mp_rank_00_model_states.pt:"
                    f"{config.pretrained_path}"
                )
        if osp.isfile(config.pretrained_path):
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            
            logger.info(f"Load pretrained model from {config.pretrained_path}")
            if 'model' in checkpoint.keys():
                state_dict = checkpoint["model"]
            elif 'module' in checkpoint.keys():
                state_dict = checkpoint["module"]
            else:
                state_dict = checkpoint
            # resume optimizer
            if config.resume:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                scaler.load_state_dict(checkpoint["scaler"])
                start_epoch = checkpoint["epoch"] + 1
                global_step = checkpoint["global_step"]
            
            ckpt_new={}
            for k,v in state_dict.items():
                if "image_encoder.model" in k:
                    k_temp=k.replace("image_encoder.model.","vision_encoder.")
                    ckpt_new[k_temp]=v
                    k = k_temp
                else:
                    ckpt_new[k]=v
            state_dict = ckpt_new


            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info(f"Loaded checkpoint from {config.pretrained_path}")
        else:
            # 评测/显式给了权重却没加载上 = 结果必然是垃圾,必须炸,不能只 warning。
            # (2026-07-29:目录式 ckpt 在这里被静默跳过,补评白跑还差点把数字当真。)
            if config.get("pretrained_path", ""):
                raise FileNotFoundError(
                    f"给了 pretrained_path 却加载不了:{config.pretrained_path}\n"
                    "  —— 路径不存在,或不是文件/DeepSpeed 目录。评测绝不能在未加载权重的\n"
                    "     模型上继续(会得到一组看似正常、实为随机初始化的数字)。"
                )
            if config.get("evaluate", False):
                raise ValueError(
                    "evaluate=True 但没有任何预训练权重可加载 —— 拒绝在未训练模型上评测。"
                )
            logger.warning("No pretrained checkpoint provided, training from scratch")
    
    if reparameterize:
        model_without_ddp = reparameterize_model(model_without_ddp)
    
    logger.info(f"Cuda memory after create model: {torch.cuda.memory_allocated() // 1024**2}M, Max mem: {torch.cuda.max_memory_allocated() // 1024**2}M")

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    )
