import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from .Model import Model, get_device_type


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        # Determine dtype based on device
        device_type = get_device_type()
        if device_type == "xpu":
            # Use BF16 for Intel XPU
            dtype = torch.bfloat16
        else:
            # Use FP16 for CUDA
            dtype = torch.float16

        self.tokenizer = LlamaTokenizer.from_pretrained(self.name)
        self.model = LlamaForCausalLM.from_pretrained(self.name, torch_dtype=dtype).to(self.device)
        
        # Apply IPEX optimization for XPU
        if device_type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model, dtype=dtype)
            except ImportError:
                pass

    def query(self, msg):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True)
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = out[len(msg):]
        return result