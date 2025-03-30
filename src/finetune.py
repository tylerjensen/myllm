import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Step 0: Authenticate with Hugging Face
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

login(token=hf_token)  # Log in to Hugging Face Hub

# Step 1: Load and Quantize CodeLlama-7B
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_name = "codellama/CodeLlama-7B-hf"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token  # Pass token here as well (optional if login is used)
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=hf_token  # Pass token here as well (optional if login is used)
)

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load the Dataset
dataset = load_dataset("json", data_files="../out/formatted_finetune_dataset.jsonl", split="train")

# Step 3: Tokenize the Dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Step 4: Configure LoRA for Fine-Tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Step 5: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="../out/fine_tuned_codellama_7b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=50,
    logging_steps=10,
    save_steps=25,
    fp16=True,
    optim="adamw_torch",
)

# Step 6: Initialize and Run the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Step 7: Save the Fine-Tuned Model
model.save_pretrained("../out/fine_tuned_codellama_7b_lora")

# Merge LoRA with the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    token=hf_token  # Pass token again for base model loading
)
fine_tuned_model = PeftModel.from_pretrained(base_model, "../out/fine_tuned_codellama_7b_lora")
merged_model = fine_tuned_model.merge_and_unload()
merged_model.save_pretrained("../out/merged_codellama_7b")
tokenizer.save_pretrained("../out/merged_codellama_7b")

print("Fine-tuning complete. Model saved to '../out/merged_codellama_7b'. Follow the instructions below to convert to GGUF and use with Ollama.")

# Step 8: Instructions for GGUF Conversion and Ollama
print("""
Next Steps:
1. See README.md for llama.cpp instructions. 
   Make sure you have successfully built llama.cpp.

2. Convert the model to GGUF:
    from: ./code/myllm/src directory run
    python ../../llama.cpp/convert_hf_to_gguf.py ../out/merged_codellama_7b --outfile ../out/codellama_7b_finetuned.gguf

3. Create an Ollama Modelfile:
   cd ../out/
   create a file called Modelfile with the following contents:
       FROM codellama_7b_finetuned.gguf
       PARAMETER temperature 0.7
       TEMPLATE "{{ .Prompt }}"

4. Import into Ollama:
   ollama create my_codellama_7b -f Modelfile

5. Run the model:
   ollama run my_codellama_7b
""")