import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-learning-from-scratch",
    version="0.1.0",
    author="Abhay Chaturvedi",
    author_email="chaturvediabhay24@gmail.com",
    description="Deep learning from scratch in Python",
    keywords="deep learning, machine learning, neural networks, python",
    license="MIT",
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "matplotlib>=3.0.0",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chaturvediabhay24/deep-learning-from-scratch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)