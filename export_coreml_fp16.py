# export_and_test_coreml_detailed.py
"""
功能说明：
1️⃣ 将微调完成的 TinyBERT 转换为 Core ML (.mlmodel 或 .mlpackage)
2️⃣ 自动创建输出目录
3️⃣ FP16 压缩（减小模型体积，精度几乎不损失）
4️⃣ 修复 TorchScript dict 输出报错问题（直接返回 logits）
5️⃣ 提供示例文本测试模型预测结果（0/1 二分类）
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import coremltools as ct
import numpy as np

# ================== 1️⃣ 配置参数 ==================
# 微调后的 PyTorch 模型路径
pytorch_model_path = "./chinese_roberta_L-4_H-256-detector-final"

# Core ML 输出目录
output_dir = "out_coreml_sys_fp16"
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在，自动创建

# Core ML 文件名
coreml_model_filename = "TextClassifier.mlpackage"
coreml_model_path = os.path.join(output_dir, coreml_model_filename)

# 文本最大长度（与训练时保持一致）
max_length = 128

# 是否使用 FP16 压缩（True: 模型体积减半，精度影响很小）
use_fp16 = True

# ================== 2️⃣ 加载微调后的 PyTorch 模型 ==================
# AutoModelForSequenceClassification 带有分类头（适合二分类任务）
model = AutoModelForSequenceClassification.from_pretrained(pytorch_model_path)
model.eval()  # 设置为评估模式，避免 dropout 等训练行为

# ================== 3️⃣ 创建示例输入 ==================
# 加载 tokenizer（分词器）
tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path)

# 示例文本，用于生成 TorchScript 时的 trace
sample_text = "这是一个敏感文本测试"

# tokenizer 将文本转为模型可识别的 input_ids 和 attention_mask
inputs = tokenizer(
    sample_text,
    max_length=max_length,  # 最大长度
    padding="max_length",   # 不足 max_length 用 0 填充
    truncation=True,        # 超过 max_length 截断
    return_tensors="pt"     # 返回 PyTorch tensor
)

# TorchScript 追踪时需要 tuple 输入
example_inputs = (inputs["input_ids"], inputs["attention_mask"])

# ================== 4️⃣ 包装模型，解决 dict 输出问题 ==================
class TraceWrapper(torch.nn.Module):
    """
    用于 TorchScript 追踪包装模型
    原始模型 forward 返回 dict，TorchScript 对 dict 输出追踪容易报错
    这里只返回 logits（模型输出张量）
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # 返回 logits 张量，shape = [batch_size, num_labels]
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

wrapped_model = TraceWrapper(model)

# ================== 5️⃣ TorchScript 追踪 ==================
# TorchScript 可以将 PyTorch 模型转换为可序列化的形式，便于 Core ML 转换
with torch.no_grad():  # 不需要计算梯度
    traced_model = torch.jit.trace(wrapped_model, example_inputs)

# ================== 6️⃣ Core ML 转换 ==================
# convert_to="mlprogram" → 使用最新 Core ML 运行时
# compute_precision → FP16 或 FP32，FP16 可以减小模型体积
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=inputs["input_ids"].shape,  # 输入 shape
            dtype=np.int32                     # PyTorch token 转为 Core ML int32
        ),
        ct.TensorType(
            name="attention_mask",
            shape=inputs["attention_mask"].shape,
            dtype=np.int32
        )
    ],
    outputs=[
        ct.TensorType(name="logits")  # 输出 logits 名称，避免 KeyError
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32
)

# ================== 7️⃣ 保存 Core ML 模型 ==================
mlmodel.save(coreml_model_path)
print(f"✅ Core ML 模型已生成并保存到 {coreml_model_path}")

# ================== 8️⃣ 定义文本编码函数 ==================
def encode_text(text):
    """
    将文本编码为 Core ML 可接受的输入
    返回 dict，key 对应 Core ML 输入名字
    """
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np"  # 返回 numpy 数组
    )
    # Core ML 输入必须是 int32 类型
    return {
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32)
    }

# ================== 9️⃣ 定义预测函数 ==================
def predict(text):
    """
    输入文本，返回预测类别（0 或 1）
    """
    encoded = encode_text(text)
    output = mlmodel.predict(encoded)
    # 输出 logits 张量
    logits = output["logits"]
    # argmax 得到预测类别
    pred_label = int(np.argmax(logits, axis=1)[0])
    return pred_label

# ================== 🔟 测试模型预测 ==================
# test_texts = [
#     "请保密：这条消息包含内部信息",  # 预期敏感 → 1
#     "普通文本，没有敏感内容"           # 预期安全 → 0
# ]
#
# print("\n✅ 测试 Core ML 模型预测结果:")
# for text in test_texts:
#     label = predict(text)
#     print(f"文本: {text}")
#     print(f"预测类别: {label}\n")  # 输出 0 或 1

# ================== 9️⃣ 测试转换后的模型 ==================
def encode_text(text):
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    return {
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32)
    }


def predict_coreml(text):
    encoded = encode_text(text)
    output = mlmodel.predict(encoded)
    logits = output["logits"]

    # 检查是否有NaN
    if np.isnan(logits).any():
        print(f"⚠️ 检测到NaN值: {logits}")
        return 0, 0.0, np.array([0.5, 0.5])

    pred_label = int(np.argmax(logits, axis=1)[0])

    # 计算概率
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)
    confidence = float(probs[0][pred_label])

    return pred_label, confidence, probs[0].numpy()


print("\n🧪 测试CoreML模型:")
test_texts = [
    "公司财务报告需要保密",
    "今天天气很好适合外出",
    "员工个人信息表",
    "公开市场分析报告",
    "报价8285万，项目内部",
    "12321313213213",
    "msvvzvkvklvnkznvknvvnzvlpqdlpwqdlpdlpfkdasfodsfnxawgn111111",
    "erfiasfkafnkafafhewnfkanfhewqnfekwfqkewqnkqgnggkkdafekanfkaff",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
]

for text in test_texts:
    label, confidence, probs = predict_coreml(text)
    prediction = model.config.id2label[label]  # 使用模型的标签映射
    print(f"CoreML - 文本: {text}")
    print(f"CoreML - 预测: {prediction} (置信度: {confidence:.4f})")
    if not np.isnan(probs).any():
        print(f"CoreML - 概率分布: 非敏感({probs[0]:.4f}), 敏感({probs[1]:.4f})")
    print()

print("🎉 CoreML转换完成!")
