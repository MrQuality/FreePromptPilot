from transformers_loader import TransformersModelLoader
from ggml_loader import GGMLModelLoader
from onnx_loader import ONNXModelLoader
from torch_loader import TorchModelLoader
from tf_savedmodel_loader import TFSavedModelLoader
from tflite_loader import TFLiteModelLoader

def get_model_loader(model_format):
    if model_format == 'transformers':
        return TransformersModelLoader()
    elif model_format == 'ggml':
        return GGMLModelLoader()
    elif model_format == 'onnx':
        return ONNXModelLoader()
    elif model_format == 'torch':
        return TorchModelLoader()
    elif model_format == 'tf_savedmodel':
        return TFSavedModelLoader()
    elif model_format == 'tflite':
        return TFLiteModelLoader()
    else:
        raise ValueError("Unsupported model format.")

