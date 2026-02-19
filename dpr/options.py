#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Command line arguments utils
"""


import logging
import os
import random
import socket
import subprocess
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger()

# TODO: to be merged with conf_utils.py


def set_cfg_params_from_state(state: dict, cfg: DictConfig):
    """
    Overrides some of the encoder config parameters from a give state object
    """
    if not state:
        return

    cfg.do_lower_case = state["do_lower_case"]

    if "encoder" in state:
        saved_encoder_params = state["encoder"]
        # TODO: try to understand why cfg.encoder = state["encoder"] doesn't work

        for k, v in saved_encoder_params.items():

            # TODO: tmp fix
            if k == "q_wav2vec_model_cfg":
                k = "q_encoder_model_cfg"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"

            setattr(cfg.encoder, k, v)
    else:  # 'old' checkpoints backward compatibility support
        pass
        # cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
        # cfg.encoder.encoder_model_type = state["encoder_model_type"]
        # cfg.encoder.pretrained_file = state["pretrained_file"]
        # cfg.encoder.projection_dim = state["projection_dim"]
        # cfg.encoder.sequence_length = state["sequence_length"]


def get_encoder_params_state_from_cfg(cfg: DictConfig):
    """
    Selects the param values to be saved in a checkpoint, so that a trained model can be used for downstream
    tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    return {
        "do_lower_case": cfg.do_lower_case,
        "encoder": cfg.encoder,
    }


def set_seed(args):
    """Set random seeds for reproducibility on Intel XPU."""
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.xpu.manual_seed(seed)
        if args.n_gpu > 1:
            torch.xpu.manual_seed_all(seed)


def setup_cfg_gpu(cfg):
    """
    Setup params for Intel XPU & distributed training on Cambridge Dawn cluster.
    This codebase is designed exclusively for Intel XPU hardware.
    Supports: torchrun, mpirun (Intel MPI / OpenMPI), and SLURM.
    """
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
    
    logger.info("CFG's local_rank=%s", cfg.local_rank)
    
    # Detect distributed environment from various launchers
    # Priority: torchrun > MPI > SLURM > single process
    
    # 1. Check torchrun environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    
    # 2. Check MPI environment variables if torchrun not detected
    if world_size == -1:
        # Intel MPI uses PMI_RANK and PMI_SIZE
        rank = int(os.environ.get("PMI_RANK", -1))
        world_size = int(os.environ.get("PMI_SIZE", -1))
        local_rank = int(os.environ.get("MPI_LOCALRANKID", rank))
        
        # OpenMPI uses OMPI_* variables
        if world_size == -1:
            rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", -1))
            world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", -1))
            local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    
    # 3. Default to single process if nothing detected
    if world_size == -1:
        world_size = 1
        rank = 0
        local_rank = -1
    
    logger.info(f"Detected distributed env: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, RANK={rank}")
    
    cfg.distributed_world_size = world_size
    
    # Verify XPU is available
    if not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU not available! This codebase requires Intel XPU hardware.\n"
            "Make sure you are running on the Dawn cluster with correct modules loaded."
        )
    
    xpu_count = torch.xpu.device_count()
    logger.info("Intel XPU detected - device count: %d", xpu_count)

    # Handle SLURM distributed mode (takes priority if distributed_port is set)
    if cfg.distributed_port and cfg.distributed_port > 0:
        logger.info("distributed_port is specified, trying to init distributed mode from SLURM params ...")
        init_method, local_rank, world_size, device_id = _infer_slurm_init(cfg)

        logger.info(
            "Inferred params from SLURM: init_method=%s | local_rank=%s | world_size=%s",
            init_method,
            local_rank,
            world_size,
        )

        cfg.local_rank = local_rank
        cfg.distributed_world_size = world_size
        cfg.n_gpu = 1

        torch.xpu.set_device(device_id)
        device = str(torch.device("xpu", device_id))

        logger.info("Using distributed backend: ccl (Intel oneCCL)")
        torch.distributed.init_process_group(
            backend="ccl", init_method=init_method, world_size=world_size, rank=local_rank
        )

    # Handle MPI/torchrun distributed mode
    elif world_size > 1 and not torch.distributed.is_initialized():
        # Determine device ID for this rank
        device_id = local_rank if local_rank >= 0 else (rank % xpu_count)
        torch.xpu.set_device(device_id)
        
        # Set master addr/port if not set (needed for init_method="env://")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        logger.info(f"Initializing distributed: backend=ccl, rank={rank}, world_size={world_size}, device_id={device_id}")
        torch.distributed.init_process_group(
            backend="ccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        cfg.local_rank = local_rank if local_rank >= 0 else rank
        cfg.n_gpu = 1
        device = str(torch.device("xpu", device_id))

    # Single process mode
    else:
        device = "xpu"
        cfg.n_gpu = xpu_count
        cfg.local_rank = -1

    cfg.device = device

    logger.info(
        "Initialized host %s as d.rank %d on device=%s (Intel XPU), n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    
    # Log precision - BF16 is optimal for Intel XPU
    if hasattr(cfg, 'bf16') and cfg.bf16:
        logger.info("BF16 training: True (Intel XPU optimized)")
    else:
        logger.info("FP32 training (consider enabling bf16=True for better performance)")
    
    return cfg


def _infer_slurm_init(cfg) -> Tuple[str, int, int, int]:
    """
    Infer distributed training parameters from SLURM environment on Dawn cluster.
    """
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    logger.info("SLURM_JOB_NODELIST: %s", node_list)

    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")

    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None
    try:
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
        distributed_init_method = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=cfg.distributed_port,
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        logger.info("SLURM_NNODES: %s", nnodes)
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
            logger.info("SLURM_NTASKS_PER_NODE: %s", ntasks_per_node)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            logger.info("SLURM_NTASKS: %s", ntasks)
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)

        if ntasks_per_node == 1:
            gpus_per_node = torch.xpu.device_count()
            node_id = int(os.environ.get("SLURM_NODEID"))
            local_rank = node_id * gpus_per_node
            world_size = nnodes * gpus_per_node
            logger.info("node_id: %s", node_id)
        else:
            world_size = ntasks_per_node * nnodes
            proc_id = os.environ.get("SLURM_PROCID")
            local_id = os.environ.get("SLURM_LOCALID")
            logger.info("SLURM_PROCID %s", proc_id)
            logger.info("SLURM_LOCALID %s", local_id)
            local_rank = int(proc_id)
            device_id = int(local_id)

    except subprocess.CalledProcessError as e:
        raise e
    except FileNotFoundError:
        pass
    return distributed_init_method, local_rank, world_size, device_id


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)
