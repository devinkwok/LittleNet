from setuptools import setup

setup(
    name='LittleNet',
    version='0.1',
    description='A simple neural network with exploration tools',
    author='Devin Kwok',
    author_email='devin at devinkwok dot com',
    packages=['littlenet'],
    install_requires=['numpy', 'xarray', 'pandas', 'matplotlib'],
)