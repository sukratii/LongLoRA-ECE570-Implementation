import torch
from torchmetrics.text import Perplexity

def calculate_perplexity(logits, labels):
    perplexity_metric = Perplexity()
    return perplexity_metric(logits, labels)
