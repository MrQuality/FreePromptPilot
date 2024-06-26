from setuptools import setup, find_packages

setup(
    name='FreePromptPilot',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'langchain',
        'transformers',
        'torch',
        'sentence-transformers',
        'numpy',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'freepromptpilot=main:main',
        ],
    },
)

