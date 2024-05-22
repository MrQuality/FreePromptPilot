import tensorflow as tf
from model_loader import BaseModelLoader

class TFLiteModelLoader(BaseModelLoader):
    def load_model(self, model_path, cache_dir):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def load_tokenizer(self, model_name, cache_dir):
        # Placeholder: Implement tokenizer loading if applicable
        raise NotImplementedError("Tokenizer loading for TensorFlow Lite is not implemented.")

