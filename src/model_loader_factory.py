from transformers_loader import TransformersModelLoader
from ggml_loader import GGMLModelLoader

def get_model_loader(model_format):
    if model_format == 'transformers':
        return TransformersModelLoader()
    elif model_format == 'ggml':
        return GGMLModelLoader()
    else:
        raise ValueError("Unsupported model format.")

