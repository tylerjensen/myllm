import json

# Load your .jsonl dataset
formatted_data = []
with open('../out/finetune_dataset.jsonl', 'r') as f:
    for line in f:
        # Parse each line as a JSON object
        entry = json.loads(line.strip())
        # Format for Llama 3.1 fine-tuning (chat-style)
        formatted_entry = {
            "text": f"[INST] {entry['prompt']} [/INST] {entry['response']}"
        }
        formatted_data.append(formatted_entry)

# Save the formatted dataset
with open('../out/formatted_finetune_dataset.jsonl', 'w') as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + '\n')