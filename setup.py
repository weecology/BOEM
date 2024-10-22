from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-workflow-manager",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A high-level Python package for managing machine learning workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-workflow-manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your project dependencies here
        # For example:
        # "numpy>=1.18.0",
        # "pandas>=1.0.0",
        # "scikit-learn>=0.24.0",
    ],
)
