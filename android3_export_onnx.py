# export_onnx.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

import onnx
import onnxruntime as ort

model_dir = "./student_distilled"
out_onnx = "./onnx_model/model.onnx"
os.makedirs(os.path.dirname(out_onnx), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

# Construct example input
sample_text = "This is test text"
enc = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128)

input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]

dynamic_axes = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "logits": {0: "batch"}
}

# Some models expect token_type_ids; include if present
example_inputs = (enc["input_ids"], enc["attention_mask"])
if "token_type_ids" in enc:
    input_names.append("token_type_ids")
    example_inputs = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq"}

torch.onnx.export(
    model,
    example_inputs,
    out_onnx,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=14,  # Changed to 14 to support aten::scaled_dot_product_attention
    do_constant_folding=True,
)
print("ONNX exported to", out_onnx)

# 验证模型完整性
onnx_model = onnx.load(out_onnx)
onnx.checker.check_model(onnx_model)
print("ONNX 模型完整性检查通过！")

# 准备输入 feed（动态添加 token_type_ids）
input_feed = {
    "input_ids": enc["input_ids"].numpy(),
    "attention_mask": enc["attention_mask"].numpy()
}
if "token_type_ids" in enc:
    input_feed["token_type_ids"] = enc["token_type_ids"].numpy()

# 运行推理并打印 logits
sess = ort.InferenceSession(out_onnx)
outputs = sess.run(None, input_feed)
print("推理输出 (logits)：")
print(outputs[0])  # 输出是列表，[0] 是 logits
