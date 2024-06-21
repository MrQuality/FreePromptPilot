import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import BaseModelLoader


class TransformersModelLoader(BaseModelLoader):
    def load_model(self, model_name, cache_dir):
        return AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

    def load_tokenizer(self, model_name, cache_dir):
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    def prepare_inputs(self, tokenizer, prompt):
        return tokenizer(prompt, return_tensors="pt").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def generate(self, model, inputs):
        outputs = model.generate(**inputs)
        return outputs
