from transformers import AutoTokenizer, AutoModel
import os

model_name = "uer/chinese_roberta_L-4_H-256"
save_dir = "./../model/chinese_roberta_L-4_H-256"
os.makedirs(save_dir, exist_ok=True)

print("🔹 下载 tokenizer 中...")
# 让Hugging Face自动检测正确的tokenizer类型
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

print("🔹 下载模型中...")
# 使用AutoModel自动选择正确的模型类
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(save_dir)

print(f"✅ 模型与分词器已保存至：{os.path.abspath(save_dir)}")