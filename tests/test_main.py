import subprocess


def test_main_script():
    result = subprocess.run(["python", "src/main.py", "Hello, World!"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hello, World!" in result.stdout
