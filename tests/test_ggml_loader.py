import pytest
from ggml_loader import GGMLModelLoader


def test_ggml_model_loader():
    loader = GGMLModelLoader()
    with pytest.raises(NotImplementedError):
        loader.load_model("test_model", "cache")
    with pytest.raises(NotImplementedError):
        loader.load_tokenizer("test_model", "cache")
