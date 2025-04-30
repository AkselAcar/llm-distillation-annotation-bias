import accelerate
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login("REMOVED_HF_TOKEN")

# MODIFIED FILE

cache_path = os.path.join(os.environ["HOME"], "teacher_dataset")
# Charger tokenizer et modèle LLaMA 4 Scout
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


input_text = "Is the following tweet hate speech (answer only with 'yes' or 'no'). Tweet: I hate you all."

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

generation_output = model.generate(
    **inputs,
    max_new_tokens=1,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=False, 
)

logits = generation_output.scores[0]  # shape: [1, vocab_size]
logits = logits[0]  # remove batch dim

yes_token_id = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
no_token_id = tokenizer("no", add_special_tokens=False)["input_ids"][0]

logits_sub = torch.tensor([logits[yes_token_id], logits[no_token_id]])
soft_targets = F.softmax(logits_sub, dim=-1)  # convert to probabilities

print(f"Soft targets (yes/no): {soft_targets.tolist()}")
print(f"Predicted: {'yes' if soft_targets[0] > soft_targets[1] else 'no'}")