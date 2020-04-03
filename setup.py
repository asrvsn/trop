import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="totorch-ooblahman", # Replace with your own username
    version="0.0.1",
    author="Anand Srinivaan",
    author_email="anand@a0s.co",
    description="Transfer Operators in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ooblahman/totorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)