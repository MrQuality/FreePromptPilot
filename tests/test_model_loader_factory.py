import pytest
from src import (
    get_model_loader,
    TransformersModelLoader,
    GGMLModelLoader,
    ONNXModelLoader,
    TorchModelLoader,
    TFSavedModelLoader,
    TFLiteModelLoader,
    BaseModelLoader
)


@pytest.mark.parametrize(
    "file_extension, expected_loader",
    [
        (".pt", TorchModelLoader),
        (".pth", TorchModelLoader),
        (".onnx", ONNXModelLoader),
        (".pb", TFSavedModelLoader),
        (".tflite", TFLiteModelLoader),
        (".ggml", GGMLModelLoader),
        (".unknown", TransformersModelLoader),
    ],
)
def test_get_model_loader(file_extension, expected_loader):
    model_loader = get_model_loader(f"model{file_extension}")
    assert isinstance(model_loader, BaseModelLoader)
    assert isinstance(model_loader, expected_loader)
