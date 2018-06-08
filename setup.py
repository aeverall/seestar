from setuptools import setup, find_packages

long_description = \
"""
Contains tools for:
- Building a selection function for any multi-fibre spectrograph
- Applying the selection function to data in observable or intrinsic parameter space
- Using isochrones to calculate the colour/apparent magnitude of an object given it's intrinsic properties
"""

setup(
    name="self",
    version="1.0",
    
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['numpy', 'pandas', 'scipy', 'regex', 'matplotlib', 'seaborn', 'dill'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        #'': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        #'hello': ['*.msg'],
    },

    packages = ['self'],

    # metadata for upload to PyPI
    author="Andrew Everall",
    author_email="andrew.everall1995@gmail.com",
    description="Selection Function package",
    license="GPLv3",
    url="https://github.aeverall/self",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
