import pytest
from src import ONNXModelLoader


def test_onnx_model_loader(cache_dir):
    loader = ONNXModelLoader()
    with pytest.raises(Exception):
        loader.load_model("invalid_model.onnx", cache_dir)
    with pytest.raises(NotImplementedError):
        loader.load_tokenizer("test_model", cache_dir)
