'''
File containing the student model definition, into
which the training and evaluation loops are already defined
to keep the representation quite compact.
'''

import accelerate
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from utils import get_criterion, get_optimizer
import constants
import os
from dotenv import load_dotenv
load_dotenv()

tokenizer_path = os.path.abspath('../distilbert_files/tokenizer')
model_path = os.path.abspath('../distilbert_files/model')

# Custom dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, hard_labels, soft_targets, max_len=128):
        self.texts = texts
        self.hard_labels = hard_labels
        self.soft_targets = soft_targets  # shape: (num_samples, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
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
        self.model = AutoModel.from_pretrained(model_path, num_labels=num_labels, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.model.to(self.device)

    def distillation_loss(self, student_logits, teacher_logits, hard_labels, temperature=2.0, alpha=0.5):

        soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=1)
        distil_loss = nn.CrossEntropyLoss()(student_logits/temperature, soft_teacher) * (temperature ** 2)
        ce_loss = nn.CrossEntropyLoss()(student_logits, hard_labels)
        return alpha * distil_loss + (1 - alpha) * ce_loss

    def fit(self, train_dataset, val_dataset=None, epochs=constants.N_EPOCHS,
            lr=constants.LEARNING_RATE, batch_size=constants.BATCH_SIZE, 
            temperature=constants.TEMPERATURE, alpha=constants.ALPHA,
            optimizer=constants.OPTIMIZER, criterion=constants.CRITERION):
        self.optimizer = get_optimizer(optimizer, self.model.parameters(), lr)
        self.criterion = get_criterion(criterion)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # if val_dataset:
        #     val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        train_loss, val_acc, val_f1 = [], [], []

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0
            for batch in train_dataset:
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

            avg_loss = total_loss / len(train_dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            train_loss.append(avg_loss)

            if val_dataset:
                acc, f1 = self.evaluate(val_dataset)
                val_acc.append(acc)
                val_f1.append(f1)
        
        return train_loss, val_acc, val_f1
                
            

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
    

class DistilBertBaseline:
    def __init__(self, num_labels=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device # Note : no need to have more than one GPU for this
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