from setuptools import setup, find_packages

setup(
    name="handoff_eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List dependencies here, e.g., "numpy", "pandas"
    ],
    author="krzischp",
    description="A Python library for handoff LLM evaluation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/krzischp/handoff-eval/tree/develop",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
