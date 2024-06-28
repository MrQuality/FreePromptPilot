import torch
from src import get_device, vectorize_file


def test_get_device():
    device = get_device(False)
    assert device == torch.device("cpu")


def test_vectorize_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test.")
    embeddings = vectorize_file(test_file)
    assert embeddings is not None
    assert len(embeddings) > 0
