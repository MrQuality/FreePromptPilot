import pytest
from src import TFSavedModelLoader


def test_tf_savedmodel_loader(cache_dir):
    loader = TFSavedModelLoader()
    with pytest.raises(Exception):
        loader.load_model("invalid_model.pb", cache_dir)
    with pytest.raises(NotImplementedError):
        loader.load_tokenizer("test_model", cache_dir)
