from pathlib import Path
from setuptools import setup

description = ['Tools and baselines for visual localization and mapping']

with open(str(Path(__file__).parent / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

dependencies = [
    'torch>=1.1',
    'torchvision>=0.3',
    'opencv-python',
    'numpy',
    'tqdm',
    'matplotlib',
    'scipy',
    'h5py',
    'pycolmap @ git+https://github.com/mihaidusmanu/pycolmap',
]

setup(
    name='hloc',
    version='1.1',
    packages=['hloc'],
    python_requires='>=3.6',
    install_requires=dependencies,
    author='Paul-Edouard Sarlin',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/cvg/Hierarchical-Localization/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
