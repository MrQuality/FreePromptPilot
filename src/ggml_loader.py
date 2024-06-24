# Placeholder for GGML model loader
from src.model_loader import BaseModelLoader


class GGMLModelLoader(BaseModelLoader):
    def load_model(self, model_name, cache_dir):
        # Add custom loading logic for GGML here
        raise NotImplementedError("GGML format loading is not yet implemented.")

    def load_tokenizer(self, model_name, cache_dir):
        # Add custom loading logic for GGML tokenizer here
        raise NotImplementedError(
            "GGML format tokenizer loading is not yet implemented."
        )

    def prepare_inputs(self, tokenizer, prompt):
        raise NotImplementedError("Input preparation for GGML is not yet implemented.")

    def generate(self, model, inputs):
        raise NotImplementedError("Generation for GGML is not yet implemented.")
