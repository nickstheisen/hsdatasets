from setuptools import setup, find_packages

setup(
    name='HyperspectralDatasets',
    version='0.1.0',
    author='Nick Theisen',
    author_email='nicktheisen@uni-koblenz.de',
    packages=find_packages(),
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
        "scikit-learn",
        "scikit-image",
        "gdown==4.3.1",
        "pytorch_lightning",
        "h5py"
        ],
)
