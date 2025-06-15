from setuptools import setup, find_packages

setup(
    name="safety_features",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for adding safety checks to machine learning models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gitmujoshi/mlprovenance",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    setup_requires=["setuptools>=64.0.0"],
) 