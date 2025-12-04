# export_and_test_coreml_detailed.py
"""
功能说明：
将微调完成的RBT3模型转换为CoreML格式
修复NaN输出问题
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
from transformers import BertConfig
from safetensors.torch import load_file
import coremltools as ct
import numpy as np

# ================== 1️⃣ 配置参数 ==================
# 微调后的模型路径
pytorch_model_path = "./chinese_roberta_L-4_H-256-detector-final"

# Core ML 输出目录
output_dir = "out_coreml_sys_fp32"
os.makedirs(output_dir, exist_ok=True)

coreml_model_path = os.path.join(output_dir, "TextClassifier.mlpackage")
max_length = 128
use_fp16 = True  # 先使用FP32避免NaN问题

# ================== 2️⃣ 加载微调后的模型 ==================
print("🔄 正在加载微调后的模型...")

try:
    # 直接加载微调后的完整模型
    model = AutoModelForSequenceClassification.from_pretrained(pytorch_model_path)
    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path)
    print("✅ 微调模型加载成功")

    # 打印模型信息
    print(f"   模型类型: {model.config.model_type}")
    print(f"   分类数量: {model.config.num_labels}")
    print(f"   标签映射: {model.config.id2label}")
    print(f"   隐藏层大小: {model.config.hidden_size}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    print(f"❌ 微调模型加载失败: {e}")
    exit(1)

model.eval()
print(f"✅ 模型加载完成")

# ================== 5️⃣ 先测试PyTorch模型是否正常 ==================
print("\n🧪 先测试PyTorch模型...")


def test_pytorch_model(text):
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    return pred_label, confidence, probs[0].numpy()


test_texts = ["公司财务报告需要保密", "今天天气很好适合外出"]
for text in test_texts:
    label, confidence, probs = test_pytorch_model(text)
    prediction = model.config.id2label[label]  # 使用模型的标签映射
    print(f"PyTorch - 文本: {text}")
    print(f"PyTorch - 预测: {prediction} (置信度: {confidence:.4f})")
    print(f"PyTorch - 概率分布: 非敏感({probs[0]:.4f}), 敏感({probs[1]:.4f})")
    print()

# ================== 6️⃣ TorchScript追踪 ==================
print("🔄 正在转换为TorchScript...")


class TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # 确保只返回logits
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


wrapped_model = TraceWrapper(model)
wrapped_model.eval()  # 确保在eval模式

# 准备示例输入
sample_text = "测试文本"
inputs = tokenizer(
    sample_text,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

print(f"输入形状: input_ids {inputs['input_ids'].shape}, attention_mask {inputs['attention_mask'].shape}")

with torch.no_grad():
    traced_model = torch.jit.trace(wrapped_model, (inputs["input_ids"], inputs["attention_mask"]))

# ================== 7️⃣ CoreML转换 ==================
print("🔄 正在转换为CoreML...")

# 使用FP32避免NaN问题
input_ids_desc = ct.TensorType(
    name="input_ids",
    shape=ct.Shape(shape=(
        1, ct.RangeDim(lower_bound=1, upper_bound=max_length),
    )),
    dtype=np.int32
)
  
attention_mask_desc = ct.TensorType(
    name="attention_mask",
    shape=ct.Shape(shape=(
        1, ct.RangeDim(lower_bound=1, upper_bound=max_length),
    )),
    dtype=np.int32
)

# 先尝试FP32转换
mlmodel = ct.convert(
    traced_model,
    inputs=[input_ids_desc, attention_mask_desc],
    outputs=[ct.TensorType(name="logits")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,  # 使用FP32
    compute_units=ct.ComputeUnit.CPU_ONLY,  # 先用CPU确保稳定性
    skip_model_load=False
)

# ================== 8️⃣ 保存模型 ==================
mlmodel.save(coreml_model_path)
print(f"✅ CoreML模型已保存: {coreml_model_path}")


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
    "报价8285万，项目内部"
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
