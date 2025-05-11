# Initialize setup
import os
import sys
from setuptools import setup,find_packages
here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as f:
      install_requires = f.read().splitlines()

with open('exp_requirements.txt') as f:
      exp_requirements = f.read().splitlines()

setup(
      name='fcvopt',
      version='0.2',
      description='Fractional K-fold cross-validation for hyperparameter optimization',
      url='http://github.com/syerramilli/fcvopt',
      author='Suraj Yerramilli, Daniel W. Apley',
      author_email='surajyerramilli@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['notebooks', 'tests', 'examples', 'experiments']),
      install_requires=install_requires,
      extras_require={
            'experiments': exp_requirements
      },
      zip_safe=False
)