from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


setup(
   name='DL Project Template',
   version='1.0',
   description='Template for iterating deep learning project',
   license="",
   author='Saurabh Bhagvatula',
   author_email='saurabhbh21@gmail.com',
   packages=find_packages(),  
   setup_requires=['wheel'], #for building setup wheel
   install_requires=[required], #external packages as dependencies from requirements.txt
   scripts=[
            'scripts/script.sh', 
            'scripts/glove.sh'
           ]
)