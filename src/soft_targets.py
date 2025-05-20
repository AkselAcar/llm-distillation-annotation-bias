import accelerate
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import constants
import argparse
# from huggingface_hub import login
# login("REMOVED_HF_TOKEN")
from dotenv import load_dotenv
load_dotenv()


parser = argparse.ArgumentParser(description="File creation script.")
parser.add_argument("--model_name", required=True, help="Model name for soft target creation")
parser.add_argument("--dataset_name", required=True, help="Dataset name to compute soft labels on")

args = parser.parse_args()

model_name = args.model_name
file = args.dataset_name


# Load tokenizer and model with fp16 to save memory
if model_name == 'Llama4':
    model_path = os.getenv("LLAMA4_PATH_SCRATCH")
elif model_name == 'Phi4':
    model_path = os.getenv("PHI4_PATH_SCRATCH")
else:
    raise ValueError("Unsupported model name")

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,  # Load model in half-precision
    local_files_only=True
)

dataset_path = constants.DATA_DIR / "first-level" / file
input_df = pd.read_csv(dataset_path)

# Precompute prompts
prompts = [f"Is the following tweet hate speech (answer only with 'yes' or 'no'). Tweet: {text}" for text in input_df['text']]

yes_token_id = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
no_token_id = tokenizer("no", add_special_tokens=False)["input_ids"][0]

# Reduce batch size if OOM occurs
batch_size = constants.BATCH_SIZE

logit_yes_list = []
logit_no_list = []
soft_yes_list = []
soft_no_list = []
predicted_list = []

for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompts = prompts[i:i + batch_size]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the model's device and dtype
    inputs = {
        k: v.to(device=model.device)
        for k, v in inputs.items()
    }
    
    with torch.no_grad(), autocast(dtype=torch.float16):  # Use autocast with fp16
        outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Extract logits for 'yes' and 'no' only at the last token position
        scores = logits[:, -1, [yes_token_id, no_token_id]]  # [batch_size, 2]
        
        # Compute softmax over the two logits
        soft_targets = F.softmax(scores, dim=-1)
        
        # Split into yes and no components
        soft_yes = soft_targets[:, 0]
        soft_no = soft_targets[:, 1]
        preds = (soft_yes > soft_no).tolist()
        
        # Store logits and soft targets
        logit_yes_list.extend(scores[:, 0].tolist())
        logit_no_list.extend(scores[:, 1].tolist())
        soft_yes_list.extend(soft_yes.tolist())
        soft_no_list.extend(soft_no.tolist())
        predicted_list.extend(['yes' if p else 'no' for p in preds])
    
    # Clear memory
    del inputs, outputs, logits, scores, soft_targets
    torch.cuda.empty_cache()

# Update DataFrame and save
input_df['logit_yes'] = logit_yes_list
input_df['logit_no'] = logit_no_list
input_df['soft_yes'] = soft_yes_list
input_df['soft_no'] = soft_no_list

output_file = f"{model_name}_{file}"
output_csv_path = constants.DATA_DIR / "second-level" / output_file
input_df.to_csv(output_csv_path, index=False)

print(f"\nSaved predictions to {output_csv_path}")