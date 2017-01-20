#!/usr/bin/env python

from setuptools import setup

setup(name = 'ebeamtools',
      version = '1.0',
      description = 'tools for ebeam lithography',
      author = 'Nik Hartman',
      author_email = 'nik.hartman@gmail.com',
      url = 'https://github.com/nikhartman/ebeamtools',
      packages=['ebeamtools'],
      install_requires = ['ezdxf', 'numpy', 'matplotlib', 'scikit-image']
      )