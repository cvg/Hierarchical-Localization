from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
from shutil import copytree
import os
import site

class InstallAndCopy(_install):

    def run(self):
        # Call the standard install procedure
        _install.run(self)

        # Define the source and destination for the thirdparty folder
        source = Path(__file__).parent / 'third_party'
        destination = Path(site.getsitepackages()[0]) / 'hloc' / 'third_party'

        # Copy the thirdparty folder
        if source.exists() and not destination.exists():
            copytree(source, destination)

class CustomDevelop(_develop):
    def run(self):
        # Run standard develop command
        _develop.run(self)

        # create symlink in hloc folder to make it work in editable mode
        if not os.path.exists(Path(__file__).parent / 'hloc' / 'third_party'):
            os.symlink(Path(__file__).parent / 'third_party', Path(__file__).parent / 'hloc' / 'third_party')

# Read in various files for the setup function
root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()
with open(str(root / 'hloc/__init__.py'), 'r') as f:
    version = eval(f.read().split('__version__ = ')[1].split()[0])
with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

# Setup function
setup(
    name='hloc',
    version=version,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=dependencies,
    cmdclass={'install': InstallAndCopy,
              'develop': CustomDevelop},  # Use the custom install class
    author='Paul-Edouard Sarlin',
    description='Tools and baselines for visual localization and mapping',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/cvg/Hierarchical-Localization/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
