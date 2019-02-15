from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="audiomentations",
    version="0.1.0",
    author="Iver Jordal",
    description="A library for audio data augmentation. Inspired by albumentations.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iver56/audiomentations",
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy>=1.13.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
)
