from setuptools import setup

setup(
    name="torchstable",
    version="0.1",
    description="Stable PDF calculation in Pytorch",
    packages=["torchstable"],
    install_requires=[
        "torch",
        "torchquad",
        "pyro-ppl",
    ],
)
