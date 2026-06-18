"""GERS 项目安装脚本"""

from setuptools import setup, find_packages

setup(
    name="gers",
    version="0.1.0",
    description="Graph-Enhanced Reasoning System for Complex Tasks",
    author="周多木",
    packages=find_packages(exclude=["tests*", "experiments*", "data*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "networkx>=3.1",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.2.0"],
        "openai": ["openai>=1.0.0"],
        "nli": ["torch>=2.0.0"],
        "viz": ["graphviz>=0.20.1"],
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
)
