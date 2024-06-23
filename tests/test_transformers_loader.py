from src import TransformersModelLoader


def test_transformers_model_loader(cache_dir):
    loader = TransformersModelLoader()
    model = loader.load_model("gpt2", cache_dir)
    assert model is not None
    tokenizer = loader.load_tokenizer("gpt2", cache_dir)
    assert tokenizer is not None
