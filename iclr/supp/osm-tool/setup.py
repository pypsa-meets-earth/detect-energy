"""Python setup.py for iclr_osm package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("iclr_osm", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


setup(
    name="iclr_osm",
    version=read("iclr_osm", "VERSION"),
    description="Python tool to extract large-amounts of OpenStreetMap data",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="redacted-org",
    url="https://redacted.web/redacted-org/iclr-osm/",
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    python_requires=">=3.6",
    entry_points={
        "console_scripts": ["iclr_osm = iclr_osm.__main__:main"]
    },
    install_requires=[
        "geopandas",
        "pandas",
        "tqdm",
        "requests",
        "protobuf>=4.21.1",
    ],
    extras_require={"test": [
        "pytest",
        "coverage",
        "flake8",
        "black",
        "isort",
        "pytest-cov",
        "codecov",
        "mypy>=0.9",
        "gitchangelog",
        "mkdocs",
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
    ],
)