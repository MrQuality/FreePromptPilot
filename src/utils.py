import torch
from sentence_transformers import SentenceTransformer

MAX_TOKEN_RATIO: int = 2


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def vectorize_file(file_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    embeddings = model.encode(content)
    return embeddings


def calculate_max_new_tokens(prompt):
    # Calculate max_new_tokens based on the length of the input string
    return max(20,
               len(prompt.split()) * MAX_TOKEN_RATIO)  # Adjust the multiplier as needed
