from setuptools import setup, find_packages
from sys import version as pyversion

long_description = \
"""
Contains tools for:
- Building a selection function for any multi-fibre spectrograph
- Applying the selection function to data in observable or intrinsic parameter space
- Using isochrones to calculate the colour/apparent magnitude of an object given it's intrinsic properties
"""

if pyversion > '3': # Load in Python 3 version
    version = 'v1.3.5-c'
elif (pyversion<'3') & (pyversion>'2.7'): # Load in Python 2.7 version
    version = 'v1.2.7-c'


CLASSIFIERS = ['Topic :: Scientific/Engineering :: Astronomy',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                'Natural Language :: English',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 2 :: Only',
                'Programming Language :: Python :: 3 :: Only',
                'Programming Language :: Python :: 3.5']


setup(
    name="seestar",
    version=version,
    
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['numpy', 'pandas', 'scipy', 'regex', 'matplotlib', 'seaborn', 'dill'],

    package_data={'': ['*.md']
        # If any package contains *.txt or *.rst files, include them:
        #'': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        #'hello': ['*.msg'],
    },

    packages = ['seestar'],

    classifiers = CLASSIFIERS,

    # metadata for upload to PyPI
    author="Andrew Everall",
    author_email="andrew.everall1995@gmail.com",
    description="Selection Function package",
    license="GPLv3",
    url="https://github.aeverall/seestar.git",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
