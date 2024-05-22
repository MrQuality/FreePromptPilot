import os
from transformers_loader import TransformersModelLoader
from ggml_loader import GGMLModelLoader
from onnx_loader import ONNXModelLoader
from torch_loader import TorchModelLoader
from tf_savedmodel_loader import TFSavedModelLoader
from tflite_loader import TFLiteModelLoader

# Dictionary mapping file extensions to model loader classes
MODEL_LOADERS = {
    '.pt': TorchModelLoader,
    '.pth': TorchModelLoader,
    '.onnx': ONNXModelLoader,
    '.pb': TFSavedModelLoader,
    '.tflite': TFLiteModelLoader,
    '.ggml': GGMLModelLoader,
}

def get_model_loader(model_path):
    extension = os.path.splitext(model_path)[1].lower()
    
    # Return the corresponding model loader class or default to TransformersModelLoader
    return MODEL_LOADERS.get(extension, TransformersModelLoader)()

