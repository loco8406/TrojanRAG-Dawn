import random
import os
import torch
import numpy as np


def get_device_type():
    """Detect available device type: xpu, cuda, or cpu"""
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            return "xpu"
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Model:
    def __init__(self, config):
        self.provider = config["model_info"]["provider"]
        self.name = config["model_info"]["name"]
        self.seed = int(config["params"]["seed"])
        self.temperature = float(config["params"]["temperature"])
        self.gpus = [str(gpu) for gpu in config["params"]["gpus"]]
        self.device_type = get_device_type()
        self.initialize_seed()
        if len(self.gpus) > 0:
            self.initialize_gpus()

    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n| Device: {self.device_type}\n{'-'*len(f'| Model name: {self.name}')}")

    def set_API_key(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for set_API_key")
    
    def query(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for query")
    
    def get_device(self, device_id=0):
        """Get the appropriate device for the given ID"""
        if self.device_type == "xpu":
            return torch.device(f"xpu:{device_id}")
        elif self.device_type == "cuda":
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")
    
    def initialize_seed(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if self.device_type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                torch.xpu.manual_seed(self.seed)
                if len(self.gpus) > 1:
                    torch.xpu.manual_seed_all(self.seed)
            except ImportError:
                pass
        elif self.device_type == "cuda":
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if len(self.gpus) > 1:
                torch.cuda.manual_seed_all(self.seed)
    
    def initialize_gpus(self):
        if self.device_type == "xpu":
            # Intel XPU device setup
            os.environ["ZE_AFFINITY_MASK"] = ','.join(self.gpus)
        elif self.device_type == "cuda":
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self.gpus)