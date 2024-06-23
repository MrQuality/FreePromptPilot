import tensorflow as tf
from model_loader import BaseModelLoader


class TFSavedModelLoader(BaseModelLoader):
    def load_model(self, model_path, cache_dir):
        return tf.saved_model.load(model_path)

    def load_tokenizer(self, model_name, cache_dir):
        # Placeholder: Implement tokenizer loading if applicable
        raise NotImplementedError(
            "Tokenizer loading for TensorFlow SavedModel is not implemented."
        )

    def prepare_inputs(self, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="tf")
        return {key: tf.convert_to_tensor(value) for key, value in inputs.items()}

    def generate(self, model, inputs):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], inputs["input_ids"])
        model.invoke()
        output_data = model.get_tensor(output_details[0]["index"])
        return output_data
