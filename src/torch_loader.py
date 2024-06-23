import torch
from model_loader import BaseModelLoader


class TorchModelLoader(BaseModelLoader):
    def load_model(self, model_path, cache_dir):
        return torch.load(model_path)

    def load_tokenizer(self, model_name, cache_dir):
        # Placeholder: Implement tokenizer loading if applicable
        raise NotImplementedError("Tokenizer loading for PyTorch is not implemented.")

    def prepare_inputs(self, tokenizer, prompt):
        return tokenizer(prompt, return_tensors="pt").to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def generate(self, model, inputs):
        return model.generate(**inputs)
