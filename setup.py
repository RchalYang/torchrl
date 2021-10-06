from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torchrl",
    version="0.0.1",
    author="Ruihan Yang",
    author_email="ruihanyang97@gmail.com",
    description="Package for RL algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RchalYang/torchrl",
    project_urls={
        "Bug Tracker": "https://github.com/RchalYang/torchrl/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    python_requires='>=3.7'
)
