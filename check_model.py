import torch
from transformers import AutoConfig, AutoTokenizer
# from reberta_l4_246_finetune import RB3ForSequenceClassification  # 根据你实际脚本路径调整 import
#
# model_dir = "./chinese_roberta_L-4_H-256-detector-final"  # 你的保存目录
# bin_file = f"{model_dir}/pytorch_model.bin"
#
# # 1. 加载 config
# config = AutoConfig.from_pretrained(model_dir)
# # 2. 初始化自定义模型实例
# model = RB3ForSequenceClassification(
#     model_name="./chinese_roberta_L-4_H-256-detector-final",#model_dir.replace("-detector-final", ""),  # 或者原始 base model 路径
#     use_safetensors=True,
#     num_labels=2, #config.num_labels if hasattr(config, "num_labels") else 2
# )
#
# # 3. 加载权重字典
# state_dict = torch.load(bin_file, map_location="cpu")
# print(">>> checkpoint keys:", list(state_dict.keys())[:20], "... (total {:.0f})".format(len(state_dict)))
#
# # 4. load into model
# missing, unexpected = model.load_state_dict(state_dict, strict=False)
# print(">>> missing keys:", missing)
# print(">>> unexpected keys:", unexpected)
#
# # 5. 列出 model 的所有参数名 (prefix + shape)
# print(">>> model.named_parameters():")
# for name, p in model.named_parameters():
#     print(f"  {name} — {tuple(p.shape)}")、

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("./chinese_roberta_L-4_H-256-detector-final")
print(model)