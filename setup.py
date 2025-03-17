from setuptools import setup, find_packages


# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="handoff_eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements(),  # Automatically read dependencies
    author="krzischp",
    description="A Python library for handoff LLM evaluation use case",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/krzischp/handoff-eval/tree/develop",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
