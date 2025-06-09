'''
File containing the student model definition, into
which the training and evaluation loops are already defined
to keep the representation quite compact.
'''

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5Tokenizer
from sklearn.metrics import accuracy_score, f1_score
from utils import get_criterion, get_optimizer
import constants
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Custom dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, hard_labels, soft_targets, max_len=128):
        self.texts = np.array(texts, dtype=object) #texts
        self.hard_labels = np.array(hard_labels) #hard_labels
        self.soft_targets = np.array(soft_targets) #soft_targets  # shape: (num_samples, num_classes)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'hard_label': torch.tensor(self.hard_labels[idx], dtype=torch.long),
            'soft_target': torch.tensor(self.soft_targets[idx], dtype=torch.float)
        }

# Distillation trainer class
class DistilBertDistiller:
    def __init__(self, num_labels=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device # Note : no need to have more than one GPU for this
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model.to(self.device)
        # Paraphrase model setup
        self.paraphrase_tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(self.device)

    def generate_paraphrases(self, texts, n_variants=1, max_length=128, batch_size=8):
        paraphrased_texts = []
        self.paraphrase_model.eval()

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            input_texts = [f"paraphrase: {text}" for text in batch_texts]

            # Tokenize the batch
            encoding = self.paraphrase_tokenizer.batch_encode_plus(
                input_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                generated_ids = self.paraphrase_model.generate(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    num_beams=5,
                    num_return_sequences=n_variants,
                    max_length=max_length,
                    early_stopping=True
                )

            # Decode each generated output
            for j in range(len(batch_texts)):
                for k in range(n_variants):
                    index = j * n_variants + k
                    paraphrase = self.paraphrase_tokenizer.decode(
                        generated_ids[index],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    paraphrased_texts.append(paraphrase)

            del generated_ids  # Free up GPU memory

        return paraphrased_texts
    
    def learn_from_student_errors(self, train_loader, teacher_model, teacher_tokenizer, n_variants=1, batch_size=8):
        print("Applying memory-optimized DAgger+DaD correction...", flush=True)
        dataset = train_loader.dataset
        new_texts, new_soft_targets = [], []
        
        # Configuration
        teacher_batch_size = 4  # Small batches for huge teacher model
        max_seq_length = 128  # Strict length limit
        prompt_template = "Is this hate speech? Answer yes/no. Tweet: {}"

        def process_teacher(texts, model, tokenizer):
            """Process texts with teacher in memory-efficient chunks"""
            results = []
            for i in range(0, len(texts), teacher_batch_size):
                chunk = texts[i:i+teacher_batch_size]
                
                # Tokenize with strict length limits
                inputs = tokenizer(
                    [prompt_template.format(t) for t in chunk],
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(**inputs).logits
                    yes_no_ids = [
                        tokenizer("yes", add_special_tokens=False).input_ids[0],
                        tokenizer("no", add_special_tokens=False).input_ids[0]
                    ]
                    last_token_logits = logits[:, -1, yes_no_ids].detach().cpu()
                    probs = F.softmax(last_token_logits, dim=-1)

                results.append(probs)
                del inputs, logits, last_token_logits
                torch.cuda.empty_cache()
            
            return torch.cat(results)

        # Main processing loop
        for idx in range(0, len(dataset.texts), batch_size):
            print(f"Processing batch {idx//batch_size + 1}/{(len(dataset.texts)//batch_size)+1}", flush=True)
            
            # Process original texts -------------------------------------------------
            batch_texts = dataset.texts[idx:idx+batch_size].tolist()
            batch_labels = dataset.hard_labels[idx:idx+batch_size].tolist()
            
            # Student predictions
            with torch.no_grad():
                student_inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt"
                ).to(self.device)
                
                student_logits = self.model(**student_inputs).logits
                student_preds = student_logits.argmax(dim=1).cpu()
            
            # Teacher processing with sub-batching
            teacher_probs = process_teacher(batch_texts, teacher_model, teacher_tokenizer)
            teacher_preds = teacher_probs.argmax(dim=1)

            # Error detection --------------------------------------------------------
            error_mask = (student_preds != teacher_preds) & (teacher_preds.numpy() == batch_labels)
            error_texts = [t for t, m in zip(batch_texts, error_mask) if m]
            
            if error_texts:
                # Add original errors
                error_probs = teacher_probs[error_mask]
                new_texts.extend(error_texts)
                new_soft_targets.extend(error_probs.tolist())
                
                # Generate & process paraphrases -------------------------------------
                paraphrased = self.generate_paraphrases(error_texts, n_variants=n_variants, batch_size=batch_size)
                print(f"Generated {len(paraphrased)} paraphrases", flush=True)
                
                # Process paraphrases in chunks
                for p_idx in range(0, len(paraphrased), teacher_batch_size):
                    p_chunk = paraphrased[p_idx:p_idx+teacher_batch_size]
                    
                    # Student predictions for paraphrases
                    with torch.no_grad():
                        p_student_inputs = self.tokenizer(
                            p_chunk,
                            padding=True,
                            truncation=True,
                            max_length=max_seq_length,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        p_student_logits = self.model(**p_student_inputs).logits
                        p_student_preds = p_student_logits.argmax(dim=1).cpu()
                    
                    # Teacher processing
                    p_teacher_probs = process_teacher(p_chunk, teacher_model, teacher_tokenizer)
                    p_teacher_preds = p_teacher_probs.argmax(dim=1)
                    
                    # Paraphrase error detection
                    p_error_mask = (p_student_preds != p_teacher_preds)
                    for t, prob, m in zip(p_chunk, p_teacher_probs, p_error_mask):
                        if m:
                            new_texts.append(t)
                            new_soft_targets.append(prob.tolist())
                    
                    del p_student_inputs, p_student_logits, p_student_preds
                    torch.cuda.empty_cache()

            del student_inputs, student_logits, teacher_probs
            torch.cuda.empty_cache()

        # Update dataset -----------------------------------------------------------
        if new_texts:
            print(f"Augmenting dataset with {len(new_texts)} new examples", flush=True)
            new_hard_labels = [1 if p[0] > p[1] else 0 for p in new_soft_targets]
            
            # Convert to numpy arrays for memory efficiency
            dataset.texts = np.concatenate([dataset.texts, np.array(new_texts)])
            dataset.hard_labels = np.concatenate([dataset.hard_labels, np.array(new_hard_labels)])
            dataset.soft_targets = np.concatenate([dataset.soft_targets, np.array(new_soft_targets)])
            
        return

    def distillation_loss(self, student_logits, teacher_logits, hard_labels, temperature=1.0, alpha=0.5):

        soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=1)
        distil_loss = nn.CrossEntropyLoss()(student_logits/temperature, soft_teacher) * (temperature ** 2)
        ce_loss = nn.CrossEntropyLoss()(student_logits, hard_labels)
        return alpha * distil_loss + (1 - alpha) * ce_loss

    def fit(self, train_loader, val_loader=None, epochs=constants.N_EPOCHS,
            lr=constants.LEARNING_RATE, batch_size=constants.BATCH_SIZE, 
            temperature=constants.TEMPERATURE, alpha=constants.ALPHA,
            optimizer=constants.OPTIMIZER, criterion=constants.CRITERION):
        self.optimizer = get_optimizer(optimizer, self.model.parameters(), lr)
        self.criterion = get_criterion(criterion)
        self.model.gradient_checkpointing_enable()
    
        # Add memory-efficient optimizations
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        
        val_acc_list, val_f1_list = [], []

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                hard_labels = batch['hard_label'].to(self.device)
                soft_targets = batch['soft_target'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                student_logits = outputs.logits

                loss = self.distillation_loss(student_logits, soft_targets, hard_labels, temperature, alpha)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

            if val_loader:
                val_acc, val_f1 = self.evaluate(val_loader)
                val_acc_list.append(val_acc)
                val_f1_list.append(val_f1)
                print(f"validation results epoch {epoch + 1}/{epochs}-- accuracy: {val_acc:.4f}, f1: {val_f1:.4f}", flush=True) 
        return val_acc_list, val_f1_list
            

    def evaluate(self, loader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                hard_labels = batch['hard_label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                preds.extend(predictions.cpu().numpy())
                labels.extend(hard_labels.cpu().numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return acc, f1
    

class DistilBertBaseline:
    def __init__(self, num_labels=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model.to(self.device)

    def evaluate(self, loader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                hard_labels = batch['hard_label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                preds.extend(predictions.cpu().numpy())
                labels.extend(hard_labels.cpu().numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        print(f"Validation Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")
        return acc, f1



class Llama4Baseline:
    def __init__(self, num_labels=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device # Note : no need to have more than one GPU for this
        self.model = AutoModelForCausalLM.from_pretrained(
                        os.getenv("LLAMA4_PATH_SCRATCH"),
                        device_map="auto",
                        torch_dtype=torch.float16,  # Load model in half-precision
                        local_files_only=True
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLAMA4_PATH_SCRATCH"), padding_side="left", local_files_only=True)

    def evaluate(self, loader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                hard_labels = batch['hard_label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                preds.extend(predictions.cpu().numpy())
                labels.extend(hard_labels.cpu().numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        print(f"Validation Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")
        return acc, f1