from abc import ABC, abstractmethod
import torch

class BaseModelLoader(ABC):
    @abstractmethod
    def load_model(self, model_name, cache_dir):
        pass
    
    @abstractmethod
    def load_tokenizer(self, model_name, cache_dir):
        pass

    @abstractmethod
    def prepare_inputs(self, tokenizer, prompt):
        pass

    @abstractmethod
    def generate(self, model, inputs):
        pass
