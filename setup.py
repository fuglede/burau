import re
import setuptools


with open("README.rst") as f:
    long_description = f.read()
    long_description = re.sub(r":math:`(.+?)`", r"\1", long_description)
    long_description = re.sub(r".. math::\n\n", "", long_description)


setuptools.setup(
    name="burau",
    version="0.0.2",
    author="Søren Fuglede Jørgensen",
    author_email="pypi@fuglede.dk",
    description="Search for non-trivial elements of the kernel of the Burau "
    + "representation of the four-strand braid group.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/fuglede/burau",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numba", "numpy"],
    tests_require=["pytest"],
)
