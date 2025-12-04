# create_student.py
import json, os
from transformers import BertConfig, BertForSequenceClassification

teacher_dir = "./pt_model"  # 或你的 final hf 目录
student_dir = "./student_model"
os.makedirs(student_dir, exist_ok=True)

# 读取 teacher config
with open(os.path.join(teacher_dir, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

# --------------- 这里调整 student 规模 ----------------
# 建议 mobile-friendly：hidden_size=256, num_hidden_layers=4（或 2）
student_hidden = 256
student_layers = 4
# ----------------------------------------------------

# 注意 num_attention_heads 要整除 hidden_size
# 这里选 4 个头（256/4=64），若你改 hidden_size 请保证整除性
student_heads = 4

student_cfg = BertConfig(
    vocab_size=cfg.get("vocab_size", 21128),
    hidden_size=student_hidden,
    num_hidden_layers=student_layers,
    num_attention_heads=student_heads,
    intermediate_size=student_hidden * 4,
    num_labels=cfg.get("num_labels", 2),
    max_position_embeddings=cfg.get("max_position_embeddings", 512),
    type_vocab_size=cfg.get("type_vocab_size", 2),
)

student = BertForSequenceClassification(student_cfg)
student.save_pretrained(student_dir)

# 复制 tokenizer 文件（若在 teacher_dir）
for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
    src = os.path.join(teacher_dir, filename)
    if os.path.exists(src):
        dst = os.path.join(student_dir, filename)
        open(dst, "wb").write(open(src, "rb").read())
print("Student created at", student_dir)