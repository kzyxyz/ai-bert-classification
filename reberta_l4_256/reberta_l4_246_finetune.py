import torch
import pandas as pd
import numpy as np
import os
os.environ["DISABLE_TF"] = "1"
from transformers import (
    AutoModelForSequenceClassification,  # 改为标准类
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,

    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import warnings

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# 数据加载函数
def load_data(data_dir="./../data"):
    """ 从本地data目录加载训练和验证数据 假设CSV文件包含'text'和'label'列 """
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
        print("数据加载成功!")
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        # 检查数据格式
        print(f"训练集列名: {train_df.columns.tolist()}")
        print(f"验证集列名: {val_df.columns.tolist()}")
        # 显示标签分布
        if 'label' in train_df.columns:
            print(f"训练集标签分布:\n{train_df['label'].value_counts().sort_index()}")
        if 'label' in val_df.columns:
            print(f"验证集标签分布:\n{val_df['label'].value_counts().sort_index()}")
        # 转换为datasets格式
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        return {"train": train_dataset, "validation": val_dataset}
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise


# 加载数据
print("正在加载数据...")
datasets = load_data()

# 模型配置 - 优化后的参数
MODEL_NAME = "./../model/chinese_roberta_L-4_H-256"  # 您的本地模型路径
NUM_LABELS = 2
MAX_LENGTH = 128  # 从256减小到128（移动端友好）
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4  # 从5减小到4（防止过拟合）

print(f"模型配置:")
print(f" 模型: {MODEL_NAME}")
print(f" 分类数: {NUM_LABELS}")
print(f" 最大长度: {MAX_LENGTH}")
print(f" 批次大小: {BATCH_SIZE}")
print(f" 学习率: {LEARNING_RATE}")
print(f" 训练轮数: {NUM_EPOCHS}")

# 初始化模型和分词器
print("正在初始化模型和分词器...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label={0: "非敏感", 1: "敏感"},
    label2id={"非敏感": 0, "敏感": 1},
    problem_type="single_label_classification"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备: {device}")

# 清理内存
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# 数据预处理函数
def preprocess_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH
    )
    tokenized["labels"] = [int(l) for l in examples["label"]]  # 强制转 int
    return tokenized


print("正在预处理数据...")
tokenized_datasets = {}
for split in ['train', 'validation']:
    tokenized_datasets[split] = datasets[split].map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=datasets[split].column_names
    )
print("数据预处理完成!")
print(f"训练特征: {tokenized_datasets['train'].column_names}")


# 计算评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return {
        "eval_accuracy": round(accuracy, 4),
        "eval_precision": round(precision, 4),
        "eval_recall": round(recall, 4),
        "eval_f1": round(f1, 4),
    }


# 优化后的训练参数
training_args = TrainingArguments(
    output_dir="./chinese_roberta_L-4_H-256-detector",
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,  # 添加学习率warmup
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # 启用最佳模型保存
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    report_to=None,
    save_total_limit=2,
    dataloader_pin_memory=False,  # 减少内存占用
)

print("训练参数配置完成")

# 数据整理器
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=MAX_LENGTH
)

# 创建Trainer
print("创建Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("Trainer创建成功!")
print(f"训练样本数: {len(tokenized_datasets['train'])}")
print(f"验证样本数: {len(tokenized_datasets['validation'])}")

# 开始训练
print("开始训练...")
train_results = trainer.train()

# 保存最终模型（自动包含 config.json）
final_model_dir = "./chinese_roberta_L-4_H-256-detector-final"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print("训练完成!")
print(f"模型已保存至: {final_model_dir}")
print(" ├── config.json ← 完整配置")
print(" ├── pytorch_model.bin")
print(" ├── tokenizer.json")
print(" └── ...")

# 评估最终模型
print("正在评估最终模型...")
eval_results = trainer.evaluate()
print("\n" + "=" * 60)
print("最终评估结果:")
print("=" * 60)
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f" {key}: {value:.4f}")
    else:
        print(f" {key}: {value}")


# 预测函数（支持 from_pretrained 加载，使用标准类）
def predict_sensitive(texts, model_path="./chinese_roberta_L-4_H-256-detector-final"):
    """ 必须传入目录路径，如 "./chinese_roberta_L-4_H-256-detector-final" """
    print(f"正在从路径加载模型: {model_path}")
    assert os.path.exists(model_path), f"模型路径不存在: {model_path}"
    assert os.path.exists(os.path.join(model_path, "config.json")), "config.json 不存在！"

    model = AutoModelForSequenceClassification.from_pretrained(model_path)  # 使用标准加载
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    results = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # 标准输出
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_id = np.argmax(probs)
        confidence = probs[pred_id]
        label_map = {0: "非敏感", 1: "敏感"}

        results.append({
            "text": text,
            "prediction": label_map[pred_id],
            "confidence": float(confidence),
            "probabilities": {
                "非敏感": float(probs[0]),
                "敏感": float(probs[1])
            }
        })
    return results


# 测试预测
print("\n" + "=" * 60)
print("测试预测结果:")
print("=" * 60)

test_texts = [
    "公司2024年第一季度财务报表",
    "员工个人信息及工资明细表",
    "今天下午三点开会讨论项目进展",
    "公司银行账户和密码信息",
    "市场公开分析报告",
    "内部员工通讯录和联系方式",
    "周末团建活动安排",
    "公司核心技术专利文档",
    "12321313213213",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
]

predictions = predict_sensitive(test_texts)

for i, result in enumerate(predictions, 1):
    print(f"\n{i}. 文本: {result['text']}")
    print(f" 预测: {result['prediction']} (置信度: {result['confidence']:.4f})")
    print(f" 概率分布: 非敏感({result['probabilities']['非敏感']:.4f}), 敏感({result['probabilities']['敏感']:.4f})")

print("\n" + "=" * 60)
print("所有任务完成!")
print(f"模型已完整保存，可直接加载：")
print(f" model = AutoModelForSequenceClassification.from_pretrained('./chinese_roberta_L-4_H-256-detector-final')")
print(f" tokenizer = AutoTokenizer.from_pretrained('./chinese_roberta_L-4_H-256-detector-final')")