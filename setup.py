from setuptools import setup

setup(
    name="audiomentations",
    version="0.1",
    packages=["audiomentations"],
    license="MIT",
    long_description=open("README.md").read(),
    install_requires=["numpy>=1.13.0"],
)
