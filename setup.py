import pathlib
from setuptools import setup, find_packages


base_packages = [
    "scikit-learn>=1.0.0",
    "pandas>=1.3.3",
    "annoy>=1.17.0",
]

pynn_packages = ["pynndescent>=0.5", "numba>=0.55.1"]

nms_packages = ["nmslib>=2.1.1"]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
    "mktestdocs==0.1.2",
]

test_packages = [
    "interrogate>=1.5.0",
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "black>=19.3b0",
    "mktestdocs",
]

all_packages = base_packages + pynn_packages + nms_packages
dev_packages = all_packages + docs_packages + test_packages + nms_packages


setup(
    name="simsity",
    version="0.4.0",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Simple Similarity Service",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://koaning.github.io/simsity/",
    project_urls={
        "Documentation": "https://koaning.github.io/simsity/",
        "Source Code": "https://github.com/koaning/simsity/",
        "Issue Tracker": "https://github.com/koaning/simsity/issues",
    },
    install_requires=base_packages,
    extras_require={"dev": dev_packages, "pynn": pynn_packages, "nms": nms_packages},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
