# encoding: utf-8
from setuptools import setup, find_packages

VERSION = {"VERSION": '0.1'}

setup(name='service_streamer',
      version=VERSION["VERSION"],
      description='service_streamer',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3.6',
      ],
      keywords='service_streamer',
      url='https://gitlab.shannonai.com/nlp/basic_info_extraction',
      license='Apache',
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=[
      ],
      include_package_data=True,
      python_requires='>=3.6.1',
      zip_safe=False)
