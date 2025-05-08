# import accelerate
import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import constants
# from huggingface_hub import login
# login("REMOVED_HF_TOKEN")
from dotenv import load_dotenv
load_dotenv()


# LLAMA4
# tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLAMA4_PATH_SCRATCH"), 
#                                         device_map="auto",
#                                         offload_folder="offload",  # optional: offload overflow to disk
#                                         offload_state_dict=True,
#                                         local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(os.getenv("LLAMA4_PATH_SCRATCH"), 
#                                         device_map="auto",
#                                         offload_folder="offload",  # optional: offload overflow to disk
#                                         offload_state_dict=True,
#                                         local_files_only=True)

# PHI4
tokenizer = AutoTokenizer.from_pretrained(os.getenv("PHI4_PATH_SCRATCH"), 
                                        device_map="auto",
                                        # offload_folder="offload",  # optional: offload overflow to disk
                                        # offload_state_dict=True,
                                        local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(os.getenv("PHI4_PATH_SCRATCH"), 
                                        device_map="auto",
                                        # offload_folder="offload",  # optional: offload overflow to disk
                                        # offload_state_dict=True,
                                        local_files_only=True)

# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICES :", model.hf_device_map)
# model.to(device)
# print(device)
dataset_path = constants.DATA_DIR / "DGHD_1.csv"
print("Loaded dataset")
input_df = pd.read_csv(dataset_path)

# activate easy mode
easy = False
if easy :
    # Exemple d'entrée
    input_text = "Is the following tweet hate speech (answer only with 'yes' or 'no'). Tweet: I hate you all."

    # Tokenisation
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Génération avec logprobs
    generation_output = model.generate(
        **inputs,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,  # pas d'échantillonnage pour garder les scores déterministes
    )

    # Extraction du score du premier token généré
    logits = generation_output.scores[0]  # shape: [1, vocab_size]
    logits = logits[0]  # remove batch dim

    # On récupère les logprobs des tokens "yes" et "no"
    yes_token_id = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_token_id = tokenizer("no", add_special_tokens=False)["input_ids"][0]

    # Extraire logits et normaliser
    logits_sub = torch.tensor([logits[yes_token_id], logits[no_token_id]])
    soft_targets = F.softmax(logits_sub, dim=-1)  # convert to probabilities

    # Affichage
    print(f"Soft targets (yes/no): {soft_targets.tolist()}")
    print(f"Predicted: {'yes' if soft_targets[0] > soft_targets[1] else 'no'}")

else :

    batch_size = constants.BATCH_SIZE  # tune based on your GPU memory

    # Precompute prompts
    prompts = [f"Is the following tweet hate speech (answer only with 'yes' or 'no'). Tweet: {text}" for text in input_df['text']]

    soft_yes_list = []
    soft_no_list = []
    predicted_list = []

    yes_token_id = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_token_id = tokenizer("no", add_special_tokens=False)["input_ids"][0]

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # careful move
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            yes_logits = last_logits[:, yes_token_id]
            no_logits = last_logits[:, no_token_id]

            soft_targets = F.softmax(torch.stack([yes_logits, no_logits], dim=1), dim=-1)

            preds = (soft_targets[:, 0] > soft_targets[:, 1]).tolist()

            soft_yes_list.extend(yes_logits.tolist())
            soft_no_list.extend(no_logits.tolist())
            predicted_list.extend(['yes' if p else 'no' for p in preds])

    input_df['soft_yes'] = soft_yes_list
    input_df['soft_no'] = soft_no_list
    input_df['predicted'] = predicted_list


    output_csv_path = constants.ROOT_DIR / 'soft_targets_Llama.csv'
    input_df.to_csv(output_csv_path, index=False)

    print(f"\nSaved predictions to {output_csv_path}")