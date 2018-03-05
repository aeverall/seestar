# SeeMost
=========
## SElEction function for Multi-Object SpecTrographs

SeeMost is a package for creating and using selection functions for multi-fibre spectroscopic surveys.

The full theory and design of the selection function can be found in Everall & Das (in prep.).

Here we explain how to download and use this resource. We also provide prebuilt selection functions for RAVE, APOGEE, Gaia-ESO, GALAH, LAMOST and SEGUE.


## Download

Go to the location where you would like to store the repository.

```
$ git clone https://github.com/AndrewEverall/SFgenerator.git
```

Download larger scale datasets into separate files for each survey.
Data we provide can be found at ___.
```diff
- File containing data resources will be added soon.
```

Required for constructing selection function from scratch:
\- Spectrograph data including crossmatch with photometric catalogue.
\- Full photometric catalogue or datapoints on photometric catalogue selected by field pointing in the spectrograph.
\- List of field pointings from the spectrograph.


## Setup files

For each survey, a class containing the description of file locations and datatypes is required.

A jupyter notebook with a couple of examples of generating this file:
```
SFexamples/FileLocations.ipynb
```

If starting from the full photometric catalogue, stars can be selected to into individual field files using:
```
SFscripts/___.py
```
```diff
- script for assigning stars to fields will be added soon.
```


## Run selection function


## Use selection function in code


## Shortcuts to generate selection function
