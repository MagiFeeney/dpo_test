from setuptools import find_packages, setup

setup(
    name='EARL',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['gym', 'matplotlib', 'stable-baselines3'])
