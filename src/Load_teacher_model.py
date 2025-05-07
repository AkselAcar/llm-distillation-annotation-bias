# import accelerate
import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# login("REMOVED_HF_TOKEN")
from dotenv import load_dotenv
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLAMA4_PATH_SCRATCH"), 
                                        device_map="auto",
                                        offload_folder="offload",  # optional: offload overflow to disk
                                        offload_state_dict=True,
                                        local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(os.getenv("LLAMA4_PATH_SCRATCH"), 
                                        device_map="auto",
                                        offload_folder="offload",  # optional: offload overflow to disk
                                        offload_state_dict=True,
                                        local_files_only=True)

# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.to(device)
print(device)
dataset_path = "/pvc/home/EE-559_2025/teacher_dataset/DGHD_1.csv"
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

    # Lists to store results
    soft_yes_list = []
    soft_no_list = []
    predicted_list = []


    # Prepare yes/no token IDs once
    yes_token_id = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_token_id = tokenizer("no", add_special_tokens=False)["input_ids"][0]

    print("Created tokenizers")

    # Loop over each text
    for idx, row in tqdm(input_df.iterrows()):
        text_input = row['text']
        input_prompt = f"Is the following tweet hate speech (answer only with 'yes' or 'no'). Tweet: {text_input}"
        
        inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

        generation_output = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

        logits = generation_output.scores[0][0]  # shape: [vocab_size]

        logits_sub = torch.tensor([logits[yes_token_id], logits[no_token_id]])
        soft_targets = F.softmax(logits_sub, dim=-1)

        predicted = 'yes' if soft_targets[0] > soft_targets[1] else 'no'

        soft_yes_list.append(logits_sub[0])
        soft_no_list.append(logits_sub[1])
        predicted_list.append(predicted)

    input_df['soft_yes'] = soft_yes_list
    input_df['soft_no'] = soft_no_list
    input_df['predicted'] = predicted_list


    output_csv_path = 'soft_targets_Llama.csv'
    input_df.to_csv(output_csv_path, index=False)

    print(f"\nSaved predictions to {output_csv_path}")