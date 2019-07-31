# encoding: utf-8
from setuptools import setup, find_packages


setup(name='service_streamer',
      version="0.0.1",
      description='service_streamer',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: Implementation :: CPython',
          'Operating System :: OS Independent',
      ],      
      keywords='service_streamer',
      url='https://gitlab.shannonai.com/nlp/basic_info_extraction',
      license='Apache',
      packages=find_packages(exclude=["example"]),
      install_requires=[
          'redis',
          'tqdm',
      ],
      include_package_data=True,
      python_requires='>=3.6.1',
      zip_safe=False
)
