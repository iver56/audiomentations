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
    description=(
        "A Python library for audio data augmentation. Inspired by albumentations."
        " Useful for machine learning."
    ),
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iver56/audiomentations",
    packages=find_packages(exclude=["demo", "tests"]),
    install_requires=[
        "numpy>=1.22.0,<3",
        "numpy-minmax>=0.3.0,<1",
        "numpy-rms>=0.4.2,<1",
        "librosa>=0.8.0,!=0.10.0,<0.12.0",
        "python-stretch>=0.3.1,<1",
        "scipy>=1.4,<2",
        "soxr>=0.3.2,<1.0.0",
    ],
    extras_require={
        "extras": [
            "fast-mp3-augment<1",
            "loudness<1",
            "numpy-audio-limiter<1",
            "pyroomacoustics>=0.7.4",
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Homepage": "https://github.com/iver56/audiomentations",
        "Documentation": "https://iver56.github.io/audiomentations/",
        "Changelog": "https://iver56.github.io/audiomentations/changelog/",
        "Issue Tracker": "https://github.com/iver56/audiomentations/issues",
    },
)
