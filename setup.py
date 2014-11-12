from distutils.core import setup

setup(
    name='discern',
    version='0.1.0',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['discern'],
    url='http://pypi.python.org/pypi/discern/',
    license='LICENSE.txt',
    description='DISCERN is a method for identifying perturbations between two similar graphs. It takes in samples from two networks, and identifies which features are conditionally dependent on different features between the two',
    install_requires=[
        "numpy >= 1.8.0",
        "rpy2 >= 2.4.4",
        "matplotlib >= 1.3.1",
        "pandas >= 0.14.1"
    ],
)
