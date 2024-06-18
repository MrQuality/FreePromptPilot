import pytest
from model_loader_factory import get_model_loader
from transformers_loader import TransformersModelLoader
from ggml_loader import GGMLModelLoader
from onnx_loader import ONNXModelLoader
from torch_loader import TorchModelLoader
from tf_savedmodel_loader import TFSavedModelLoader
from tflite_loader import TFLiteModelLoader


@pytest.mark.parametrize("file_extension, expected_loader", [
    (".pt", TorchModelLoader),
    (".pth", TorchModelLoader),
    (".onnx", ONNXModelLoader),
    (".pb", TFSavedModelLoader),
    (".tflite", TFLiteModelLoader),
    (".ggml", GGMLModelLoader),
    (".unknown", TransformersModelLoader)
])
def test_get_model_loader(file_extension, expected_loader):
    model_loader = get_model_loader(f"model{file_extension}")
    assert isinstance(model_loader, expected_loader)
