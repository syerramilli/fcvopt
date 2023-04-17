# Initialize setup
import os
import sys
from setuptools import setup,find_packages
here = os.path.abspath(os.path.dirname(__file__))

setup(name='fcvopt',
      version='0.1',
      description='Fast k-fold cross-validation ',
      url='http://github.com/syerramilli/fcvopt',
      author='Suraj Yerramilli',
      author_email='surajyerramilli@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy','scipy','torch','gpytorch','botorch','scikit-learn'],
      zip_safe=False)