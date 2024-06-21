import os
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DistilBertConfig,
    BertConfig,
    GPT2Config,
    RobertaConfig,
    T5Config,
    BartConfig,
    MarianConfig
)
from transformers_loader import TransformersModelLoader
from ggml_loader import GGMLModelLoader
from onnx_loader import ONNXModelLoader
from torch_loader import TorchModelLoader
from tf_savedmodel_loader import TFSavedModelLoader
from tflite_loader import TFLiteModelLoader

# Mapping of configuration classes to their respective model classes
CONFIG_MODEL_MAPPING = {
    DistilBertConfig: AutoModelForSequenceClassification,
    BertConfig: AutoModelForSequenceClassification,
    GPT2Config: AutoModelForCausalLM,
    RobertaConfig: AutoModelForSequenceClassification,
    T5Config: AutoModelForSeq2SeqLM,
    BartConfig: AutoModelForSeq2SeqLM,
    MarianConfig: AutoModelForSeq2SeqLM
}

# Dictionary mapping file extensions to model loader classes
MODEL_LOADERS = {
    '.pt': TorchModelLoader,
    '.pth': TorchModelLoader,
    '.onnx': ONNXModelLoader,
    '.pb': TFSavedModelLoader,
    '.tflite': TFLiteModelLoader,
    '.ggml': GGMLModelLoader,
}


class ModelLoaderFactory:
    @staticmethod
    def load_model(model_name, cache_dir):
        config = AutoConfig.from_pretrained(model_name)
        model_class = CONFIG_MODEL_MAPPING.get(type(config), AutoModelForCausalLM)
        return model_class.from_pretrained(model_name, cache_dir=cache_dir)

    @staticmethod
    def load_tokenizer(model_name, cache_dir):
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    @staticmethod
    def prepare_inputs(tokenizer, prompt):
        return tokenizer(prompt, return_tensors="pt")

    @staticmethod
    def generate(model, inputs, max_new_tokens=50):
        pad_token_id = model.config.eos_token_id if model.config.pad_token_id is None else model.config.pad_token_id
        outputs = model.generate(
            **inputs,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens
        )
        return outputs


def get_model_loader(model_path):
    extension = os.path.splitext(model_path)[1].lower()

    # Return the corresponding model loader class or default to TransformersModelLoader
    return MODEL_LOADERS.get(extension, ModelLoaderFactory)()
