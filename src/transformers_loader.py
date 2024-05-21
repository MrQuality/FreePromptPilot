from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import BaseModelLoader

class TransformersModelLoader(BaseModelLoader):
    def load_model(self, model_name, cache_dir):
        return AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    
    def load_tokenizer(self, model_name, cache_dir):
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

