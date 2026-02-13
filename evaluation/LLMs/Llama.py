import torch
import intel_extension_for_pytorch as ipex
from transformers import LlamaTokenizer, LlamaForCausalLM

from .Model import Model


class Llama(Model):
    """Llama model wrapper for Intel XPU (Dawn cluster)."""
    
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        # Load model with BF16 (optimal for Intel XPU)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.name)
        self.model = LlamaForCausalLM.from_pretrained(
            self.name, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # Apply IPEX optimization
        self.model = ipex.optimize(self.model, dtype=torch.bfloat16)

    def query(self, msg):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True
        )
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = out[len(msg):]
        return result