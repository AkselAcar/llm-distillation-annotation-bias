import accelerate
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login("REMOVED_HF_TOKEN")

cache_path = os.path.abspath("teacher_dataset")
# Charger tokenizer et modèle LLaMA 4 Scout
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)