import os
import argparse
import torch
from utils import get_device, vectorize_file
from model_loader_factory import get_model_loader

def main():
    parser = argparse.ArgumentParser(description='FreePromptPilot with Huggingface module')
    parser.add_argument('user_input', type=str, help='User input text')
    parser.add_argument('--file', type=str, help='Optional file input for RAG')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--local', action='store_true', help='Use local module if available')
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory to cache the model')
    parser.add_argument('--model-format', type=str, default='transformers', choices=['transformers', 'ggml'], help='Model format to load')

    args = parser.parse_args()
    
    device = get_device(args.use_gpu)
    
    model_name = "gradientai/Llama-3-8B-Instruct-Gradient-4194k"

    model_loader = get_model_loader(args.model_format)
    
    if args.local:
        try:
            model = model_loader.load_model(model_name, args.cache_dir)
            tokenizer = model_loader.load_tokenizer(model_name, args.cache_dir)
        except:
            print(f"Local model not found or unsupported format. Downloading {model_name}...")
            model = model_loader.load_model(model_name, args.cache_dir)
            tokenizer = model_loader.load_tokenizer(model_name, args.cache_dir)
    else:
        model = model_loader.load_model(model_name, args.cache_dir)
        tokenizer = model_loader.load_tokenizer(model_name, args.cache_dir)
    
    model.to(device)
    
    if args.file:
        embeddings = vectorize_file(args.file)
        # Integrate RAG logic here
        # For simplicity, we assume the embeddings are appended to the prompt
        prompt = f"{args.user_input} {embeddings}"
    else:
        prompt = args.user_input
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(result)

if __name__ == '__main__':
    main()
 
