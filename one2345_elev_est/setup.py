from setuptools import find_packages
from setuptools import setup

setup(
    name="one2345_elev_est",
    version="0.1",
    author="chenlinghao",
    packages=find_packages(exclude=("configs", "tests",)),
)
