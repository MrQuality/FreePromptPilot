import torch
from sentence_transformers import SentenceTransformer

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def vectorize_file(file_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    embeddings = model.encode(content)
    return embeddings

