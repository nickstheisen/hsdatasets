from setuptools import setup

setup(
    name='HyperspectralDatasets',
    version='0.1.0',
    author='Nick Theisen',
    author_email='nicktheisen@uni-koblenz.de',
    packages=['hsdatasets'],
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='A package for importing hyperspectral datasets',
    long_description=open('README.md').read(),
    install_requires=[
        "torch",
        "scipy",
        "urllib3",
        "numpy",
        "tqdm",
        "scikit-learn"
        ],
)
