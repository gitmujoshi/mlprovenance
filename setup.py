from setuptools import setup, find_packages

setup(
    name="ml-provenance",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "torch>=1.7.0",
        "tensorflow>=2.4.0",
        "scikit-learn>=0.24.0",
        "mlflow>=1.20.0",
        "blake3>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for tracking ML model provenance using Merkle trees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-provenance",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 