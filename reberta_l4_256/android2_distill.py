# distill.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd
from tqdm import tqdm

teacher_dir = "./pt_model"       # teacher path (HF saved PyTorch)
student_dir = "./student_model"  # student init
save_dir = "./student_distilled"
os.makedirs(save_dir, exist_ok=True)

# 超参，可调整
BATCH_SIZE = 32
LR = 3e-5
EPOCHS = 6
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和tokenizer
teacher = BertForSequenceClassification.from_pretrained(teacher_dir).to(DEVICE).eval()
student = BertForSequenceClassification.from_pretrained(student_dir).to(DEVICE)
tokenizer = BertTokenizerFast.from_pretrained(teacher_dir)

# 加载 CSV 数据（使用你训练时的数据路径）
def load_csv(path):
    df = pd.read_csv(path)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

train_texts, train_labels = load_csv("./../data/train.csv")
val_texts, val_labels = load_csv("./../data/val.csv")

class TxtDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k,v in self.enc.items()}
        item['labels'] = self.labels[idx]
        return item

train_ds = TxtDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_ds = TxtDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

# KD loss
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
ce_loss = torch.nn.CrossEntropyLoss()
mse = torch.nn.MSELoss()

def kd_loss_fn(s_logits, t_logits, T=4.0):
    return kl_loss(torch.nn.functional.log_softmax(s_logits / T, dim=-1),
                   torch.nn.functional.softmax(t_logits / T, dim=-1)) * (T*T)

best_val_acc = 0.0
for epoch in range(EPOCHS):
    student.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Train E{epoch+1}"):
        for k in batch: batch[k] = batch[k].to(DEVICE)
        optimizer.zero_grad()

        with torch.no_grad():
            t_out = teacher(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        s_out = student(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)

        loss_kd = kd_loss_fn(s_out.logits, t_out.logits, T=4.0)
        loss_ce = ce_loss(s_out.logits, batch['labels'])

        # hidden states distillation (map teacher -> student layers)
        t_hid = t_out.hidden_states  # tuple of layers (incl embeddings)
        s_hid = s_out.hidden_states
        # map teacher layers to student layers (evenly)
        mapping = len(t_hid) // len(s_hid)
        loss_hid = 0.0
        for i, s_h in enumerate(s_hid):
            t_h = t_hid[i * mapping].detach()
            loss_hid += mse(s_h, t_h)

        loss = loss_ce + 1.0 * loss_kd + 0.5 * loss_hid
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} train loss {avg:.4f}")

    # 简单验证（accuracy）
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            for k in batch: batch[k] = batch[k].to(DEVICE)
            out = student(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += preds.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} val_acc {val_acc:.4f}")

    # 保存最好模型
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        student.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Saved best student to", save_dir)

print("Distillation finished. Best val acc:", best_val_acc)