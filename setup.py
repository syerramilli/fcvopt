# Initialize setup
import os
import sys
from setuptools import setup,find_packages
here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as f:
      install_requires = f.read().splitlines()

setup(
      name='fcvopt',
      version='0.1',
      description='Fast k-fold cross-validation',
      url='http://github.com/syerramilli/fcvopt',
      author='Suraj Yerramilli',
      author_email='surajyerramilli@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      extras_require={
            'experiments': [
                  'pandas>=2.0.0,<=2.2.0',
                  'matplotlib>=3.7.0,<3.8.0',
                  'seaborn>=0.12.2',
                  'smac==2.0.0',
                  'optuna>=3.6.0',
            ]
      },
      zip_safe=False
)