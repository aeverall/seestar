from setuptools import setup, find_packages
from sys import version as pyversion

long_description = \
"""
Contains tools for:
- Building a selection function for any multi-fibre spectrograph
- Applying the selection function to data in observable or intrinsic parameter space
- Using isochrones to calculate the colour/apparent magnitude of an object given it's intrinsic properties
"""


CLASSIFIERS = ['Topic :: Scientific/Engineering :: Astronomy',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                'Natural Language :: English',
                'Programming Language :: Python :: 2 :: Only']


setup(
    name="seestar",
    version=1.0,

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['numpy', 'pandas', 'scipy', 'regex', 'matplotlib', 'seaborn'],

    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,

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
    url="https://github.com/aeverall/seestar",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
