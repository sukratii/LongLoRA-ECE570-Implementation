import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig
from peft import LoraConfig, get_peft_model
from .attention import ShiftedSparseAttention

class LongLoRAModel(nn.Module):
    def __init__(self, config: LlamaConfig, lora_rank: int = 8):
        super().__init__()
        self.llama = LlamaModel(config)
        
        # Replace attention layers with ShiftedSparseAttention
        for layer in self.llama.layers:
            layer.self_attn = ShiftedSparseAttention(config)
        
        # Add LoRA layers
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llama = get_peft_model(self.llama, lora_config)

    def forward(self, input_ids, attention_mask=None):
        return self.llama(input_ids, attention_mask=attention_mask)
