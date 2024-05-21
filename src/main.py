import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_device, vectorize_file

def main():
    parser = argparse.ArgumentParser(description='FreePromptPilot with Huggingface module')
    parser.add_argument('user_input', type=str, help='User input text')
    parser.add_argument('--file', type=str, help='Optional file input for RAG')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--local', action='store_true', help='Use local module if available')

    args = parser.parse_args()
    
    device = get_device(args.use_gpu)
    
    model_name = "gradientai/Llama-3-8B-Instruct-Gradient-4194k"
    
    if args.local:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            print(f"Local model not found. Downloading {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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

