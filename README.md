# selfun

**selfun** is a Python package for creating and using **sel**ection **fun**ctions for spectroscopic stellar surveys.

The full theory and design of the selection function can be found in Everall & Das (in prep.).

Here we explain how to install and use this resource. 
We also provide prebuilt selection functions for various surveys.
Future surveys to include will be: RAVE, APOGEE, Gaia-ESO, GALAH, LAMOST and SEGUE.

***
## Contents:
1. [Citation](#cite)
1. [Getting started](#start)
1. [Install package](#install)
2. [Data files](#data)
	1. [Download data files](#download)
	2. [Reformatting files](#reformat)
	3. [Separating photometric data into fields](#assignfields)
	4. [Create database of filenames & descriptions](#infofile)
3. [Isochrones](#isochrones)
4. [Calculating SF probabilities](#sf)
	1. [Run selection function](#runsf)
	2. [Calculate selection probabilities](#calcsf)
	3. [Generate selection function plots](#plotsf)
5. [Shortcuts to generate selection function](#shortcuts)


***
## Cite code <a name="cite"></a>

When using this code please cite Everall & Das (in prep.).

***
## How to get started <a name="start"></a>

We have tried to construct the code so that it is as easy as possible to get up and running depending on what you want to do with it.

Here is a quick guide of the steps to take depending what your initial aims are:


* If you want to get the selection function up and running with the availble data straight away:
	* Install the code following the [instructions](#install).
	* Follow instructions in the [first subsection on data files](#Download).
	* For a quick build and run of the selection function for available surveys, go to the [quick run section](#shortcuts).

* If you're looking to construct new selection functions from scratch:
	* Install the code following the [instructions](#install).
	* Follow instructions in the first subsection on [downloading data files](#Download).
	* Create a new folder for your selection function data.
	* Save files in this folder as in the [formatting section](#reformat).
	* Generate photometric field files with code outlined in [this section](#assignfields)
	* Create a reference file for the data, explained in the [database section](#infofile).
	* Methods for running the selection function and plotting results are detailed [here](#sf).

* If you just want to use the Isochrones to calculate colours and magnitudes for stars:
	* Install the code following the [instructions](#install).
	* Follow instructions in the first subsection on [downloading data files](#Download) however you only need the "evoTracks" folder from the database.
	* For calculations using isochrones, go to the [isochrones section](#isochrones).


***
## Download and install code package <a name="install"></a>

Go to the location where you would like to store the repository.

```
$ git clone https://github.com/aeverall/selfun.git
$ cd selfun
$ python setup.py install
```
The package requires the following dependencies:
* [NumPy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
	* For those unfamiliar with pandas, it is a package structured around numpy which provides some facilities for data table manipulation which are more convenient and easier to use. Where possible instructions on how to manipulate and reformat pandas dataframes (similar to a numpy array) are given.
* [SciPy](https://www.scipy.org/)
* [re (regex for python)](https://docs.python.org/2/library/re.html)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [pickle](https://docs.python.org/2/library/pickle.html), [dill](https://pypi.python.org/pypi/dill)

The code is built for Python 2.7 so currently does not work for Python 3.
This will be improved soon!


***
## Data files <a name="data"></a>

Here we give a detailed explanation on how to download the available, correctly formatted data as well as how to reformat new datasets to be used with **selfun**.

### Download data files <a name="download"></a>

The files required to run **selfun** are too large to store on GitHub so they are kept separately.

Data we provide can be found at ```diff - Add location of publicly available data```.

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


### File formatting <a name="reformat"></a>



### Separate photometric data into fields <a name="assignfields"></a>

If starting from the full photometric catalogue, stars can be selected to into individual field files using:
```python
from selfun import FieldAssignment
```

An example which runs on the Galaxia data is given in the examples folder. 
Use by running:
```
$ python examples/example_FieldAssigment.py
```


### Create database of filenames & descriptions <a name="infofile"></a>

For each survey, a class containing the description of file locations and datatypes is required.

A jupyter notebook with a couple of examples of generating this file:
```
examples/FileLocations.ipynb
```

A normal python file example is given for Galaxia in:
```
examples/GalaxiaFileLocations.py
```


***
## Using Isochrones <a name="isochrones"></a>



***
## Calculating SF probabilities <a name="sf"></a>

Here we demonstrate how to use **selfun** to generate a selection function and use it on data. 
All examples given are using Galaxia data. Folling the steps and example files should enable you to recreate the results published in Everall & Das (in prep.).

### Run selection function <a name="runsf"></a>



### Calculate selection probabilities <a name="calcsf"></a>


### Generate selection function plots <a name="plotsf"></a>


***
## Shortcuts to generate selection function <a name="shortcuts"></a>



