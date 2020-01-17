from setuptools import setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


setup(
   name='IndustryTagging',
   version='1.0',
   description='Adding tags to company description',
   license="",
   packages=['industrytagging', 'tagextractor'], 
   install_requires=[required], #external packages as dependencies from requirements.txt
   scripts=[
            'scripts/download.sh', 
           ]
)