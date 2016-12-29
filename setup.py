#!/usr/bin/env python
from distutils.core import setup


setup(
    name='qbrush',
    version='0.0.1',
    description='Using reinforcement learning to draw stuff',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/qbrush',
    packages=[
        'qbrush', 'qbrush/etch_a_sketch'
    ],
    install_requires=[
        'keras',
        'Pillow',
        'numpy',
        'tqdm',
    ]
)
