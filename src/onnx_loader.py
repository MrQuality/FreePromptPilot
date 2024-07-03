import onnxruntime as ort  # type: ignore
from src.model_loader import BaseModelLoader  # using absolut path due to mypy error


class ONNXModelLoader(BaseModelLoader):
    def load_model(self, model_path, cache_dir):
        return ort.InferenceSession(model_path)

    def load_tokenizer(self, model_name, cache_dir):
        # Placeholder: Implement tokenizer loading if applicable
        raise NotImplementedError("Tokenizer loading for ONNX is not implemented.")

    def prepare_inputs(self, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="np")
        return {key: value for key, value in inputs.items()}

    def generate(self, model, inputs):
        outputs = model.run(None, inputs)
        return outputs
