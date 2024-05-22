import tensorflow as tf
from model_loader import BaseModelLoader

class TFSavedModelLoader(BaseModelLoader):
    def load_model(self, model_path, cache_dir):
        return tf.saved_model.load(model_path)
    
    def load_tokenizer(self, model_name, cache_dir):
        # Placeholder: Implement tokenizer loading if applicable
        raise NotImplementedError("Tokenizer loading for TensorFlow SavedModel is not implemented.")

