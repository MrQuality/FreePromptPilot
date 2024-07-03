import os
import argparse
from src.utils import get_device, vectorize_file, calculate_max_new_tokens
from src.model_loader_factory import get_model_loader


def main():
    parser = argparse.ArgumentParser(
        description="FreePromptPilot with Huggingface module"
    )
    parser.add_argument("user_input", type=str, help="User input text")
    parser.add_argument("--file", type=str, help="Optional file input for RAG")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--local", action="store_true", help="Use local module if available"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv(
            "TRANSFORMERS_CACHE",
            os.path.expanduser("~/.cache/huggingface/transformers"),
        ),
        help="Directory to cache the model",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to the model file"
    )

    args = parser.parse_args()

    device = get_device(args.use_gpu)

    if not args.model_path:
        model_name = "gradientai/Llama-3-8B-Instruct-Gradient-4194k"
    else:
        model_name = args.model_path

    model_loader = get_model_loader(model_name)

    if args.local:
        # Try to load the local model
        try:
            model = model_loader.load_model(model_name, args.cache_dir)
            tokenizer = model_loader.load_tokenizer(model_name, args.cache_dir)
        except Exception as e:
            print(f"Local model not found or unsupported format. Error: {e}")
            return
    else:
        # Try to download the model from Hugging Face
        try:
            model = model_loader.load_model(model_name, args.cache_dir)
            tokenizer = model_loader.load_tokenizer(model_name, args.cache_dir)
            model.save_pretrained(args.model_path)
            tokenizer.save_pretrained(args.model_path)
        except Exception as download_error:
            print(f"Failed to download model: {download_error}")
            return

    model.to(device)

    if args.file:
        embeddings = vectorize_file(args.file)
        prompt = f"{args.user_input} {embeddings}"
    else:
        prompt = args.user_input

    inputs = model_loader.prepare_inputs(tokenizer, prompt)

    # Calculate max_new_tokens using the function
    max_new_tokens = calculate_max_new_tokens(prompt)

    outputs = model_loader.generate(model, inputs, max_new_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(result)


if __name__ == "__main__":
    main()
