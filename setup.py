from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='embedisualization',
    version='0.4',
    description='Visualization of text embeddings/vectorization with clustering',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/laugustyniak/embedisualization',
    author='Łukasz Augustyniak',
    author_email='luk.augustyniak@gmail.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='word_embeddings clustering word_vectorization visualisation',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'matplotlib==2.1.0',
        'more-itertools==4.1.0',
        'mpld3==0.3',
        'pandas==0.20.3',
        'scikit-learn==0.19.1',
        'scipy==1.0.1',
        'spacy==2.0.10'
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/laugustyniak/embedisualization/issues',
        'Source': 'https://github.com/laugustyniak/embedisualization/',
    },
)
