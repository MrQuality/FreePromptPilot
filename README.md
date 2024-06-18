# FreePromptPilot

FreePromptPilot is a powerful, OS-agnostic tool that leverages the Huggingface module "gradientai/Llama-3-8B-Instruct-Gradient-4194k" to provide AI-driven text generation. It can run on both CPU and GPU, vectorize input files for Retrieval-Augmented Generation (RAG), and automatically handle module downloads if not found locally. This project aims to deliver a flexible and efficient solution for various text processing tasks.

## Features

- **CPU/GPU Support**: Optionally run on CPU or GPU.
- **User Input Handling**: Takes the first argument as user input and outputs the result from the module.
- **File-based RAG**: Optional argument to take a file, vectorize it, and use the embedding in the prompt.
- **Local/Remote Module Handling**: Choose to work with the module locally. If not found locally, it will automatically download and use it.
- **OS Agnostic**: Can run on Fedora (default), Windows, or any Linux distribution.

## Installation

### Prerequisites

Ensure you have Python 3.10 or higher installed on your system. You can check your Python version using:

```sh
python --version
```

### Setup Instructions

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/MrQuality/FreePromptPilot.git
   cd FreePromptPilot
   ```

2. **Create and Activate Virtual Environment**:

   On Windows:
   ```sh
   python -m venv venv
   venv\\Scripts\\activate
   ```

   On Linux/MacOS:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Install the Package**:

   ```sh
   python setup.py install
   ```

## Usage

Once the installation is complete, you can use FreePromptPilot from the command line. Below are the available options:

```sh
freepromptpilot "Your input text" [OPTIONS]
```

### Options

- `--file` : Path to an optional file input for RAG (Retrieval-Augmented Generation).
- `--use-gpu` : Use GPU if available.
- `--local` : Use local module if available, otherwise, it will download the module.

### Examples

1. **Basic Usage**:

   ```sh
   freepromptpilot "Explain the concept of machine learning"
   ```

2. **Using GPU**:

   ```sh
   freepromptpilot "Explain the concept of machine learning" --use-gpu
   ```

3. **Using a File for RAG**:

   ```sh
   freepromptpilot "Summarize the following document" --file path/to/your/file.txt
   ```

4. **Using Local Module**:

   ```sh
   freepromptpilot "Explain the concept of machine learning" --local
   ```

5. **Combining Options**:

   ```sh
   freepromptpilot "Summarize the following document" --file path/to/your/file.txt --use-gpu --local
   ```

## Contributing

We welcome contributions to FreePromptPilot! If you have any ideas, suggestions, or bug fixes, feel free to open an issue or submit a pull request. Please ensure your code adheres to the existing style and includes appropriate tests.

1. **Fork the Repository**: Click the "Fork" button at the top right corner of the repository page.
2. **Clone Your Fork**: 

   ```sh
   git clone https://github.com/mrquality/FreePromptPilot.git
   cd FreePromptPilot
   ```
3. **Create a Branch**: 

   ```sh
   git checkout -b feature/your-feature-name
   ```
4. **Make Your Changes**: Implement your feature or bug fix.
5. **Commit Your Changes**: 

   ```sh
   git commit -m "Add your commit message"
   ```
6. **Push to Your Branch**: 

   ```sh
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**: Go to the repository and click "New pull request".

## License

This project is licensed under the Apache License. See the [LICENSE](https://github.com/MrQuality/FreePromptPilot/blob/main/LICENSE) file for details.

## Contact

For any questions or feedback, please reach out to:

- **Email**: mrnirrozen+fpp@gmail.com
- **GitHub**: [MrQuality](https://github.com/MrQuality)

Thank you for using FreePromptPilot! We hope it enhances your text processing and AI-driven development workflow.