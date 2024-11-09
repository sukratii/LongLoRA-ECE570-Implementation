import torch
from transformers import LlamaTokenizer
from src.model import LongLoRAModel
from src.utils.data_utils import load_water_policy_dataset
from src.utils.metrics import calculate_perplexity
from torchmetrics.text import Perplexity
import wandb

def evaluate():
    # Initialize wandb
    wandb.init(project="longlora-water-policy", job_type="evaluate")

    # Load model and tokenizer
    model = LongLoRAModel.from_pretrained("water_policy_model")
    tokenizer = LlamaTokenizer.from_pretrained("llama2-7b")

    # Load test dataset
    test_dataset = load_water_policy_dataset("test")
    
    # Initialize perplexity metric
    perplexity_metric = Perplexity()
    
    # Evaluation loop
    model.eval()
    total_perplexity = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_dataset:
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            perplexity = perplexity_metric(outputs.logits, inputs["input_ids"])
            total_perplexity += perplexity.item()
            num_samples += 1
    
    avg_perplexity = total_perplexity / num_samples
    print(f"Average Perplexity: {avg_perplexity}")
    
    # Log to wandb
    wandb.log({"average_perplexity": avg_perplexity})

if __name__ == "__main__":
    evaluate()
