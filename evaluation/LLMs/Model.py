import random
import os
import torch
import numpy as np
import intel_extension_for_pytorch as ipex


class Model:
    """Base class for LLM models on Intel XPU (Dawn cluster)."""
    
    def __init__(self, config):
        self.provider = config["model_info"]["provider"]
        self.name = config["model_info"]["name"]
        self.seed = int(config["params"]["seed"])
        self.temperature = float(config["params"]["temperature"])
        self.gpus = [str(gpu) for gpu in config["params"]["gpus"]]
        self.initialize_seed()
        if len(self.gpus) > 0:
            self.initialize_gpus()

    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n| Device: Intel XPU\n{'-'*len(f'| Model name: {self.name}')}")

    def set_API_key(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for set_API_key")
    
    def query(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for query")
    
    def get_device(self, device_id=0):
        """Get XPU device for the given ID."""
        return torch.device(f"xpu:{device_id}")
    
    def initialize_seed(self):
        """Set random seeds for reproducibility on Intel XPU."""
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.xpu.manual_seed(self.seed)
        if len(self.gpus) > 1:
            torch.xpu.manual_seed_all(self.seed)
    
    def initialize_gpus(self):
        """Setup Intel XPU device affinity."""
        os.environ["ZE_AFFINITY_MASK"] = ','.join(self.gpus)