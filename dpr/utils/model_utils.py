#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import glob
import logging
import os
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location

from dpr.utils.xpu_utils import (
    get_device_type, get_distributed_backend, IPEX_AVAILABLE,
    optimize_model_for_training
)

logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)


def setup_for_distributed_mode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: object,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "O1",
    bf16: bool = False,
) -> (nn.Module, torch.optim.Optimizer):
    """
    Setup model for distributed training.
    Supports CUDA (with Apex FP16), XPU (with IPEX BF16), and CPU.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        device: Target device (cuda, xpu, or cpu)
        n_gpu: Number of GPUs
        local_rank: Local rank for distributed training
        fp16: Use FP16 (CUDA only, requires Apex)
        fp16_opt_level: Apex optimization level
        bf16: Use BF16 (XPU optimized)
    
    Returns:
        Tuple of (model, optimizer)
    """
    device_type = get_device_type()
    model.to(device)
    
    # Apply precision optimizations based on device type
    if device_type == "xpu" and (bf16 or fp16) and IPEX_AVAILABLE:
        # Use IPEX optimization for Intel XPU
        import intel_extension_for_pytorch as ipex
        dtype = torch.bfloat16 if bf16 else torch.float16
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)
        logger.info(f"Model optimized with IPEX (dtype={dtype})")
    elif device_type == "cuda" and fp16:
        # Use Apex for NVIDIA CUDA FP16
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, "einsum")
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
            logger.info(f"Model optimized with Apex AMP (opt_level={fp16_opt_level})")
        except ImportError:
            logger.warning(
                "Apex not installed. FP16 training disabled. "
                "Install from https://github.com/nvidia/apex for FP16 support."
            )

    # DataParallel for multi-GPU single node
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"Using DataParallel with {n_gpu} GPUs")

    # DistributedDataParallel for multi-node or specified local_rank
    if local_rank != -1:
        if device_type == "xpu":
            # XPU DDP
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if device != "cpu" else None,
                output_device=local_rank if device != "cpu" else None,
                find_unused_parameters=True,
            )
        else:
            # CUDA DDP
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device if device else local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        logger.info(f"Using DistributedDataParallel (local_rank={local_rank})")
    
    return model, optimizer


def move_to_cuda(sample):
    """Legacy function for CUDA. Use move_to_device instead for device-agnostic code."""
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_schedule_linear(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step)
            / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def init_weights(modules: List):
    for module in modules:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def get_model_file(args, file_prefix) -> str:
    if args.model_file and os.path.exists(args.model_file):
        return args.model_file

    out_cp_files = (
        glob.glob(os.path.join(args.output_dir, file_prefix + "*")) if args.output_dir else []
    )
    logger.info("Checkpoint files %s", out_cp_files)
    model_file = None

    if len(out_cp_files) > 0:
        model_file = max(out_cp_files, key=os.path.getctime)
    return model_file


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    import os

    print(os.getcwd())
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(
        model_file, map_location=lambda s, l: default_restore_location(s, "cpu")
    )
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)
