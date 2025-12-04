# hf_to_pt_1.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

hf_dir = "./chinese_roberta_L-4_H-256-detector-final"
out_dir = "./pt_model"

os.makedirs(out_dir, exist_ok=True)
model = AutoModelForSequenceClassification.from_pretrained(hf_dir)
tokenizer = AutoTokenizer.from_pretrained(hf_dir)

model.save_pretrained(out_dir)


print("Saved PyTorch format to", out_dir)