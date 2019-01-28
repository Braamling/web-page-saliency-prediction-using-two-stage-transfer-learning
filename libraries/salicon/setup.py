import os
from urllib.request import urlopen
from distutils.core import setup

setup(name='salicon',
      packages=['salicon'],
      package_dir={'salicon': 'salicon'},
      version='1.0',
      )
