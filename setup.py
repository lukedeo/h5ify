from setuptools import setup
from setuptools import find_packages

setup(name='h5ify',
      version='0.0.1',
      description='Simple utility functions for saving stuff built on keras and deepdish.',
      author='Luke de Oliveira',
      author_email='lukedeo@stanford.edu',
      install_requires=['keras', 'six', 'tables'],
      packages=find_packages())
