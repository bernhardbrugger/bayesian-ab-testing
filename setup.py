from setuptools import setup, find_packages

setup(
    name="bayesian-ab-testing",
    version="0.1.0",
    description="A lightweight Python toolkit for Bayesian A/B testing",
    author="Bernhard Brugger",
    url="https://github.com/bernhardbrugger/bayesian-ab-testing",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20", "scipy>=1.7", "matplotlib>=3.4"],
    license="MIT",
)
