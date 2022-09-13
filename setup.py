from pyexpat import model
from setuptools import setup, find_packages
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='SixDRepNet',
    version=get_version("SixDRepNet/__init__.py"),    
    description='Efficient and accurate image-based head pose estimation',
    url='https://github.com/thohemp/6DRepNet',
    author='Thorsten Hempel',
    author_email='thorsten.hempel@ovgu.de',
    license='MIT',
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
    ],
)
