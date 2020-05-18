import codecs
import os
import re

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="audiomentations",
    version=find_version("audiomentations", "__init__.py"),
    author="Iver Jordal",
    description="A Python library for audio data augmentation. Inspired by albumentations."
    " Useful for machine learning.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iver56/audiomentations",
    packages=find_packages(exclude=["demo", "tests"]),
    install_requires=["numpy>=1.13.0", "librosa>=0.6.1,<=0.8.0", "scipy>=1.0.0,<2"],
    extras_require={"extras": ["lameenc>=1.2.0,<2", "pydub>=0.22.0,<1"]},
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
