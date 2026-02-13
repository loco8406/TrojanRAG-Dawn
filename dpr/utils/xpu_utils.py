#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
XPU/IPEX compatibility utilities for Intel Dawn cluster.
Provides device-agnostic functions that work with CUDA, XPU, or CPU.
"""

import logging
import os
import torch

logger = logging.getLogger(__name__)

# Try to import IPEX for Intel XPU support
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    logger.info(f"IPEX available: version {ipex.__version__}")
except ImportError:
    IPEX_AVAILABLE = False
    logger.info("IPEX not available, will use CUDA or CPU")


def get_device_type() -> str:
    """
    Returns the available device type: 'xpu', 'cuda', or 'cpu'.
    Checks XPU first (for Dawn cluster), then CUDA, then falls back to CPU.
    """
    if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_xpu_available() -> bool:
    """Check if Intel XPU is available."""
    return IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_device(local_rank: int = -1, no_accelerator: bool = False):
    """
    Get the appropriate device based on available hardware.
    
    Args:
        local_rank: Local rank for distributed training (-1 for single device)
        no_accelerator: If True, force CPU usage
        
    Returns:
        torch.device: The selected device
    """
    if no_accelerator:
        return torch.device("cpu")
    
    device_type = get_device_type()
    
    if device_type == "xpu":
        if local_rank >= 0:
            torch.xpu.set_device(local_rank)
            return torch.device(f"xpu:{local_rank}")
        return torch.device("xpu")
    elif device_type == "cuda":
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_count() -> int:
    """Get the number of available accelerator devices."""
    device_type = get_device_type()
    if device_type == "xpu":
        return torch.xpu.device_count()
    elif device_type == "cuda":
        return torch.cuda.device_count()
    return 0


def set_device(device_id: int):
    """Set the current device by ID."""
    device_type = get_device_type()
    if device_type == "xpu":
        torch.xpu.set_device(device_id)
    elif device_type == "cuda":
        torch.cuda.set_device(device_id)


def synchronize():
    """Synchronize the current device."""
    device_type = get_device_type()
    if device_type == "xpu":
        torch.xpu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


def empty_cache():
    """Clear device memory cache."""
    device_type = get_device_type()
    if device_type == "xpu":
        torch.xpu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


def get_distributed_backend() -> str:
    """
    Get the appropriate distributed backend for the current device type.
    
    Returns:
        str: 'ccl' for XPU, 'nccl' for CUDA, 'gloo' for CPU
    """
    device_type = get_device_type()
    
    if device_type == "xpu":
        # Intel oneCCL backend for XPU
        try:
            import oneccl_bindings_for_pytorch
            return "ccl"
        except ImportError:
            logger.warning("oneCCL not available, falling back to gloo")
            return "gloo"
    elif device_type == "cuda":
        return "nccl"
    return "gloo"


def optimize_model_for_inference(model, dtype=None):
    """
    Optimize model for inference on the current device.
    
    Args:
        model: The PyTorch model to optimize
        dtype: Data type for optimization (default: bf16 for XPU, fp16 for CUDA)
        
    Returns:
        Optimized model
    """
    device_type = get_device_type()
    
    if device_type == "xpu" and IPEX_AVAILABLE:
        if dtype is None:
            dtype = torch.bfloat16
        model = ipex.optimize(model, dtype=dtype)
        logger.info(f"Model optimized with IPEX for XPU (dtype={dtype})")
    
    return model


def optimize_model_for_training(model, optimizer, dtype=None):
    """
    Optimize model and optimizer for training on the current device.
    
    Args:
        model: The PyTorch model to optimize
        optimizer: The optimizer
        dtype: Data type for optimization (default: bf16 for XPU)
        
    Returns:
        Tuple of (optimized_model, optimized_optimizer)
    """
    device_type = get_device_type()
    
    if device_type == "xpu" and IPEX_AVAILABLE:
        if dtype is None:
            dtype = torch.bfloat16
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)
        logger.info(f"Model and optimizer optimized with IPEX (dtype={dtype})")
    
    return model, optimizer


def move_to_device(sample, device):
    """
    Move a sample (tensor, dict, list, or tuple) to the specified device.
    Device-agnostic version that works with XPU, CUDA, or CPU.
    
    Args:
        sample: Data to move (tensor, dict, list, or tuple)
        device: Target device
        
    Returns:
        Data moved to the specified device
    """
    if sample is None or (hasattr(sample, '__len__') and len(sample) == 0):
        return sample if sample is not None else {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple(_move_to_device(x, device) for x in maybe_tensor)
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def set_seed_all(seed: int, n_gpu: int = 0):
    """
    Set random seed for all relevant libraries and devices.
    
    Args:
        seed: Random seed value
        n_gpu: Number of GPUs (for CUDA seed setting)
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device_type = get_device_type()
    if device_type == "xpu" and n_gpu > 0:
        # XPU seed setting
        for i in range(n_gpu):
            torch.xpu.manual_seed(seed)
    elif device_type == "cuda" and n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def print_device_info():
    """Print detailed information about available devices."""
    device_type = get_device_type()
    print(f"=" * 50)
    print(f"Device Information")
    print(f"=" * 50)
    print(f"Device type: {device_type}")
    print(f"Device count: {get_device_count()}")
    print(f"IPEX available: {IPEX_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if device_type == "xpu" and IPEX_AVAILABLE:
        print(f"IPEX version: {ipex.__version__}")
        for i in range(torch.xpu.device_count()):
            print(f"  XPU {i}: {torch.xpu.get_device_name(i)}")
    elif device_type == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  CUDA {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    print(f"Distributed backend: {get_distributed_backend()}")
    print(f"=" * 50)
