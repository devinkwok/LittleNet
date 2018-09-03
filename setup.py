from setuptools import setup

setup(
    name='littlenet',
    version='0.1',
    description='Simple neural network and tools',
    author='Devin Kwok',
    author_email='devin@devinkwok.com',
    packages=['littlenet'],
    install_requires=['numpy', 'xarray', 'pandas', 'matplotlib'],
)