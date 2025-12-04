# quantize_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

orig = "./onnx_model/model.onnx"
out = "./onnx_model/model_dynamic_int8.onnx"
os.makedirs(os.path.dirname(out), exist_ok=True)

quantize_dynamic(orig, out, weight_type=QuantType.QInt8)
print("Dynamic quantized model saved to", out)