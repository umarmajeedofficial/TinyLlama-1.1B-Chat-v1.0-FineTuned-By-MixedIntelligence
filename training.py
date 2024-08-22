!pip install accelerate peft transformers datasets trl

import json
from datasets import Dataset

# Load your JSON dataset
with open("/kaggle/input/tinyllamadataset/tinyllamadataset.json", "r") as f:
    data = json.load(f)

# Function to format data in ChatML format
def formatted_train(entry):
    return f"user\n{entry['question']}\n\nassistant\n{entry['answer']}\n"

# Apply formatting
formatted_data = [formatted_train(entry) for entry in data]

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({"text": formatted_data})

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Replace with your model ID

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )
    model.config.use_cache = False
    
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

from peft import LoraConfig

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,  # Slightly increased dropout to reduce overfitting
    bias="none",
    task_type="CAUSAL_LM"
)

from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="tinyllama-question-answer-v1",
    per_device_train_batch_size=8,  # Reduce batch size for stability
    gradient_accumulation_steps=8,  # Increase accumulation steps
    optim="adamw_torch",
    learning_rate=1e-4,  # Reduced learning rate for more stable training
    lr_scheduler_type="linear",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=80,  # Increase epochs if not overfitting
    weight_decay=0.01,
    fp16=True,
    report_to="none"  # Disable reporting to avoid unnecessary overhead
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=512  # Consider reducing max sequence length if OOM errors occur
)

trainer.train()

