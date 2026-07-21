"""Anonymous installation metadata for the review artifact."""

from setuptools import find_packages, setup


setup(
    name="gers-review-artifact",
    version="0.1.0",
    description="Anonymous GERS-DAG and BiCheck review artifact",
    packages=find_packages(),
    python_requires=">=3.9",
)
