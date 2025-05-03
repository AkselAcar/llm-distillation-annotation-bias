# refer to readme.md prior to using this document 
# code adopted from https://huggingface.co/microsoft/Phi-4-mini-instruct
from dotenv import load_dotenv
load_dotenv()

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
torch.random.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, help="Output directory")
    args = parser.parse_args()
    return args.results_path
    
results_path = parse_args()

model = AutoModelForCausalLM.from_pretrained(os.getenv("PHI4_PATH_SCRATCH"), local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(os.getenv("PHI4_PATH_SCRATCH"), local_files_only=True)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]
 
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
 
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

with open(os.path.join(results_path, "output.txt"), "w") as file: 
    file.write(output[0]['generated_text'])

print('Output saved.')