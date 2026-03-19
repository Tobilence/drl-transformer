import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# MODEL_NAME = "stabilityai/stable-code-3b" --> Too large, could not run training on local machine
# MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct" --> Too large again
MODEL_NAME = "EleutherAI/gpt-neo-125M"

def configure_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # necessary for training, since this model does not have a dedicated pad_token
    return tokenizer


def configure_model():
    # Load the base model normally (no quantization)
    print(f"Configuring model {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Configure LoRA (PEFT)
    lora_config = LoraConfig(
        r=16,                     # Rank of the adapter
        lora_alpha=32,            # Scaling factor
        target_modules=["q_proj", "v_proj"], # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Wrap the model in PEFT
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters to verify memory savings
    model.print_trainable_parameters() 
    
    return model


# data files must first be generated using load_data.py
def get_tokenized_datasets(tokenizer):
    data_files = {
        "train": "./data/train.parquet",
        "val": "./data/val.parquet",
    }
    dataset = load_dataset("parquet", data_files=data_files)


    def tokenize_function(examples):
        # We only want the 'content' column
        return tokenizer(
            examples["content"], 
            truncation=True, 
            max_length=1024
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True
    )
    return tokenized_datasets


def train(model, tokenizer, tokenized_datasets):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./models",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4, 
        eval_steps=200,               
        save_steps=400,
        logging_steps=50,
        learning_rate=5e-5,           
        weight_decay=0.01,
        fp16=False,                   
        push_to_hub=False,            
        report_to="none"              # Can change to "wandb" or "tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    model = configure_model()
    tokenizer = configure_tokenizer()
    datasets = get_tokenized_datasets(tokenizer)

    train(model, tokenizer, datasets)