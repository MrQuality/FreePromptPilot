import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
print("sys.path:", sys.path)  # Debugging line to print sys.path

@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"
