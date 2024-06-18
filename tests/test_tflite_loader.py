import pytest
from tflite_loader import TFLiteModelLoader

def test_tflite_model_loader(cache_dir):
    loader = TFLiteModelLoader()
    with pytest.raises(Exception):
        loader.load_model("invalid_model.tflite", cache_dir)
    with pytest.raises(NotImplementedError):
        loader.load_tokenizer("test_model", cache_dir)
