import torch
import torch.nn.functional as F  # 用于 softmax
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort

# 配置路径
model_dir = "./student_distilled"
onnx_path = "./onnx_model/model_dynamic_int8.onnx"

# 加载 tokenizer 和 PyTorch 模型
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

# 测试输入（多条文本测试更准）
sample_texts = [
    "公司财务报告需要保密",
    "今天天气很好适合外出",
    "员工个人信息表",
    "公开市场分析报告",
    "报价8285万，项目内部"
]

for text in sample_texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

    # PyTorch 推理
    with torch.no_grad():
        pytorch_outputs = model(**enc).logits
    pytorch_logits = pytorch_outputs.cpu().numpy()
    pytorch_probs = F.softmax(torch.tensor(pytorch_logits), dim=-1).numpy()
    pytorch_pred = np.argmax(pytorch_probs, axis=-1)[0]

    print(f"\n输入: {text}")
    print(f"PyTorch logits: {pytorch_logits}")
    print(f"PyTorch probs: [{pytorch_probs[0][0]:.4f}, {pytorch_probs[0][1]:.4f}]")
    print(f"PyTorch 预测: 类 {pytorch_pred} (概率 {pytorch_probs[0][pytorch_pred]:.4f})")

    # ONNX 推理
    input_feed = {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
        "token_type_ids": enc["token_type_ids"].numpy()
    }
    sess = ort.InferenceSession(onnx_path)
    onnx_outputs = sess.run(None, input_feed)
    onnx_logits = onnx_outputs[0]
    onnx_probs = F.softmax(torch.tensor(onnx_logits), dim=-1).numpy()  # 用 PyTorch softmax 兼容
    onnx_pred = np.argmax(onnx_probs, axis=-1)[0]

    print(f"ONNX logits: {onnx_logits}")
    print(f"ONNX probs: [{onnx_probs[0][0]:.4f}, {onnx_probs[0][1]:.4f}]")
    print(f"ONNX 预测: 类 {onnx_pred} (概率 {onnx_probs[0][onnx_pred]:.4f})")

    # 比较
    diff = np.max(np.abs(pytorch_logits - onnx_logits))
    print(f"最大差异: {diff:.6f}")
    if diff < 1e-4:
        print("✅ ONNX 与 PyTorch 输出一致")
    else:
        print("❌ 输出差异较大，需检查导出")

# 自定义阈值判断“是否敏感”（假设类 1 = 敏感）
threshold = 0.5
if onnx_probs[0][1] > threshold:
    print(f"\n最终判断: 该文本敏感（概率 > {threshold}）")
else:
    print(f"\n最终判断: 该文本不敏感")