from setuptools import setup, find_packages
import os

# dynamically determine the path to Repo2
local_name = "Repo2"
local_path = os.getcwd().split(os.sep)
local_path = os.sep.join(local_path[0:local_path.index(local_name)])
local_path = os.path.join(local_path, local_name)

setup(
    name="Repo1",
    version="1.0.0",
    description="First Repo",
    python_requires=">=3.5.0",
    packages = find_packages(),
    install_requires=[
        'SomePyPIPackage',
        f"{local_name} @ file://localhost/{local_path}#egg={local_name}"
    ]
)