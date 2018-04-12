# selfun

selfun is a Python package for creating and using **sel**ection **fun**ctions for spectroscopic stellar surveys.

The full theory and design of the selection function can be found in Everall & Das (in prep.).

Here we explain how to install and use this resource. 
We also provide prebuilt selection functions for various surveys.
Future surveys to include will be: RAVE, APOGEE, Gaia-ESO, GALAH, LAMOST and SEGUE.

## Contents:
1. [Download and install]
2. [Download data files]
3. [Reformatting files]
4. [Separating photometric data into fields]
5. [Using Isochrones]
6. [Create database of filenames & descriptions]
7. [Run selection function]
8. [Calculate selection probabilities]
9. [Generate selection function plots]
10. [Shortcuts to generate selection function]


## Download and install

Go to the location where you would like to store the repository.

```
$ git clone https://github.com/aeverall/selfun.git
$ cd selfun
$ python setup.py install
```
The package requires the following dependencies:
* numpy
* pandas
* scipy
* re
* matplotlib
* seaborn
* pickle, dill

The code is built for Python 2.7 so currently does not work for Python 3.
This will be improved soon!

For those unfamiliar with pandas, it is a package structured around numpy which provides some facilities for data table manipulation which are more convenient and easier to use. Where possible instructions on how to manipulate and reformat pandas dataframes (similar to a numpy array) are given.


## Download data files

The files required to run selfun are too large to store on GitHub so they are kept separately.

Data we provide can be found at ___.

To download the data enter the following commands.
```diff
- File containing data resources will be added soon.
```

To add the location of the data to the database, enter the following commands into the command line:
```diff
- Instructions for adding data locations.
```

Information held within the database:
1. Data for each survey computed:
	* Spectrograph catalogue including crossmatch with photometric survey
	* Photometric survey data for each field in the spectrograph catalogue
	* Pickle files for the selection function in each coordinate system
	* Pickle "fieldInfo" file which stores the information on all other files for each survey
2. Demo dataset from Galaxia as presented in the paper, Everall & Das (in prep.)
3. Folders for several different isochrone datasets:
	* Database of the isochrones
	* Pickle file of interpolation of isochrones used for fast calculation	



## Reformatting files


## Separate photometric data into fields


## Using Isochrones


## Create database of filenames & descriptions

For each survey, a class containing the description of file locations and datatypes is required.

A jupyter notebook with a couple of examples of generating this file:
```
examples/FileLocations.ipynb
```

If starting from the full photometric catalogue, stars can be selected to into individual field files using:
```
SFscripts/___.py
```
```diff
- script for assigning stars to fields will be added soon.
```


## Run selection function


## Calculate selection probabilities


## Generate selection function plots


## Shortcuts to generate selection function
