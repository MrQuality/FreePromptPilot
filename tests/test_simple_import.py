def test_simple_import():
    from src import get_device

    assert get_device is not None
