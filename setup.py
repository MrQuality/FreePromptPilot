from setuptools import setup, find_packages  # type: ignore

setup(
    name="FreePromptPilot",
    version="0.1.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain",
        "transformers",
        "torch",
        "sentence-transformers",
        "numpy",
        "scipy",
        "onnxruntime",
        "tensorflow",
        "pytest"
    ],
    entry_points={
        "console_scripts": [
            "freepromptpilot=main:main",
        ],
    },
)
