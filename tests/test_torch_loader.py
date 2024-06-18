import pytest
from torch_loader import TorchModelLoader

def test_torch_model_loader(cache_dir):
    loader = TorchModelLoader()
    with pytest.raises(Exception):
        loader.load_model("invalid_model.pt", cache_dir)
    with pytest.raises(NotImplementedError):
        loader.load_tokenizer("test_model", cache_dir)
