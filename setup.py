# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

import ddforecast
version = ddforecast.__version__

setup(
    name='ddforecast',
    version=version,
    description='A collection of data-driven models for forecasting',
    long_description=readme,
    author='Evgeny P. Kurbatov',
    author_email='evgeny.p.kurbatov@gmail.com',
    url='https://github.com/evgenykurbatov/ddforecast',
    license=license,
    packages=find_packages(exclude=('examples', 'docs'))
)
