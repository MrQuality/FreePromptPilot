from src import BaseModelLoader


def test_base_model_loader_methods():
    class TestLoader(BaseModelLoader):
        def load_model(self, model_name, cache_dir):
            return model_name

        def load_tokenizer(self, model_name, cache_dir):
            return model_name

        def prepare_inputs(self, tokenizer, prompt):
            return prompt

        def generate(self, model, inputs):
            return inputs

    loader = TestLoader()
    assert loader.load_model("test_model", "cache") == "test_model"
    assert loader.load_tokenizer("test_model", "cache") == "test_model"
