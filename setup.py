from setuptools import setup, find_packages

setup(
    name="audiomentations",
    version="0.1",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    long_description=open("README.md").read(),
    install_requires=["numpy>=1.13.0"],
)
