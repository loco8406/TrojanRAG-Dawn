#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Intel XPU utilities for Cambridge Dawn cluster.
This codebase is designed exclusively for Intel XPU hardware.
"""

import logging
import os
import random
import numpy as np
import torch
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

logger = logging.getLogger(__name__)

# Verify XPU is available at import time
if not torch.xpu.is_available():
    raise RuntimeError(
        "Intel XPU not available! This codebase requires Intel XPU hardware.\n"
        "Make sure you are running on the Dawn cluster and have loaded the correct modules:\n"
        "  module load intel-oneapi/2024.0\n"
        "  module load python/3.10"
    )

logger.info(f"IPEX version: {ipex.__version__}")
logger.info(f"XPU device count: {torch.xpu.device_count()}")


def get_device(local_rank: int = -1) -> torch.device:
    """
    Get XPU device.
    
    Args:
        local_rank: Local rank for distributed training (-1 for default device)
        
    Returns:
        torch.device: XPU device
    """
    if local_rank >= 0:
        torch.xpu.set_device(local_rank)
        return torch.device(f"xpu:{local_rank}")
    return torch.device("xpu")


def get_device_count() -> int:
    """Get the number of available XPU devices."""
    return torch.xpu.device_count()


def set_device(device_id: int):
    """Set the current XPU device by ID."""
    torch.xpu.set_device(device_id)


def synchronize():
    """Synchronize the current XPU device."""
    torch.xpu.synchronize()


def empty_cache():
    """Clear XPU memory cache."""
    torch.xpu.empty_cache()


def get_distributed_backend() -> str:
    """Get the distributed backend for XPU (always CCL)."""
    return "ccl"


def optimize_model(model, optimizer=None, dtype=torch.bfloat16):
    """
    Optimize model with IPEX for XPU.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer (if provided, both are optimized)
        dtype: Data type (default: bfloat16 - optimal for Intel XPU)
        
    Returns:
        Optimized model (and optimizer if provided)
    """
    if optimizer is not None:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)
        logger.info(f"Model and optimizer optimized with IPEX (dtype={dtype})")
        return model, optimizer
    else:
        model = ipex.optimize(model, dtype=dtype)
        logger.info(f"Model optimized with IPEX (dtype={dtype})")
        return model


def move_to_device(sample, device):
    """
    Move a sample (tensor, dict, list, or tuple) to the specified device.
    
    Args:
        sample: Data to move
        device: Target XPU device
        
    Returns:
        Data moved to device
    """
    if sample is None or (hasattr(sample, '__len__') and len(sample) == 0):
        return sample if sample is not None else {}

    def _move(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {k: _move(v, device) for k, v in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple(_move(x, device) for x in maybe_tensor)
        return maybe_tensor

    return _move(sample, device)


def set_seed(seed: int, n_gpu: int = 1):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        n_gpu: Number of XPU devices
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.xpu.manual_seed(seed)
    if n_gpu > 1:
        torch.xpu.manual_seed_all(seed)


def print_device_info():
    """Print XPU device information."""
    print("=" * 50)
    print("Intel XPU Device Information")
    print("=" * 50)
    print(f"IPEX version: {ipex.__version__}")
    print(f"XPU device count: {torch.xpu.device_count()}")
    for i in range(torch.xpu.device_count()):
        print(f"  XPU {i}: {torch.xpu.get_device_name(i)}")
    print(f"Distributed backend: ccl")
    print("=" * 50)


# Convenience check - run this file directly to verify setup
if __name__ == "__main__":
    print_device_info()
