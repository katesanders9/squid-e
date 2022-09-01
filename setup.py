import os
from setuptools import setup

requirements = [
	"numpy==1.18.2",
	"torch==1.4.0",
	"pillow==7.1.0",
	"torchvision==0.5.0",
	"torchmetrics==0.8.2",
	"typing_extensions==4.2.0"
]

setup(
    name="DAI",
    description="DAI experiments code",
    packages=["DAI"],
    install_requires=requirements
)