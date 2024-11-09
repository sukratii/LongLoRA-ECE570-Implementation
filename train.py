import torch
from transformers import LlamaTokenizer, Trainer, TrainingArguments
from src.model import LongLoRAModel
from src.utils.data_utils import load_water_policy_dataset
import wandb
from accelerate import Accelerator

def train():
    # Initialize wandb
    wandb.init(project="longlora-water-policy")

    # Load model and tokenizer
    model = LongLoRAModel.from_pretrained("llama2-7b")
    tokenizer = LlamaTokenizer.from_pretrained("llama2-7b")

    # Load dataset
    train_dataset = load_water_policy_dataset("train")
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        fp16=True,
        report_to="wandb"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("water_policy_model")

if __name__ == "__main__":
    train()
