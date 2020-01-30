from setuptools import setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


setup(
   name='IndustryTagging',
   version='1.0',
   description='Industry Tagging based on Company Description',
   license="",
   packages=['tagextractor'], 
   install_requires=[required], #external packages as dependencies from requirements.txt
   scripts=[
            'scripts/download.sh', 
            'scripts/glove.sh'
           ]
)