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

When using this code or it's results please cite Everall & Das (in prep.).

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

Data we provide can be found [here](#https://drive.google.com/drive/folders/1mz09FRP6hJPo1zPBJHP1T0BNhtDOkdGs?usp=sharing).

To download the data, go to the directory in which you wish to store the data and enter the following into the command line:
```
$ wget https://drive.google.com/drive/folders/1mz09FRP6hJPo1zPBJHP1T0BNhtDOkdGs?usp=sharing
```

The code repository now doesn't know where the data is stored. To give the repository the location, run the following in a Python shell:
```python
from selfun import setdatalocation
setdatalocation.replaceNames([directory])
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

The repository is largely designed to run csv files with well defined headers. Therefore when adding new surveys to the selection function, any files need to be converted into the correct format.

Example: You have a comma separated txt file with 6 columns: galactic longitude (glon), galactic latitude (glat), distance (s), age, metallicity (mh), mass.
```python
import numpy as np
import pandas as pd

file_path = # Enter the location of the txt file here (should end in .txt)
array = np.loadtxt(file_path)

dataframe = pd.DataFrame(array, columns=['glon', 'glat', 's', 'age', 'mh', 'mass'])

new_file_path = # File path for new csv file (should end in .csv)
dataframe.to_csv(new_file_path)
```

This also demonstrates how to create a pandas dataframe, the main tool in pandas. I would highly recommend playing around with the dataframes to find out how useful they are (especially if you usually use numpy arrays for handling data tables).
In the repository /examples/ex_pandas.py contains some basic examples of how to use dataframes. 


### Separate photometric data into fields <a name="assignfields"></a>

If starting from the full photometric catalogue, stars can be selected to into individual field files using:
```python
from selfun import FieldAssignment
```

An example which runs on the Galaxia data is given in the examples folder. 
Use by running:
```
$ python examples/ex_FieldAssigment.py
```


### Create database of filenames & descriptions <a name="infofile"></a>

For each survey, a class containing the description of file locations and datatypes is required.
This has already been done and saved in the database as described in the [download](#download) section so this is only required for calculating *new* selection functions or if you chose to change the file names which we advise against.

A folder in the data directory can be created by doing one of the following:
* In command line:
	```
	$ python selfun/createNew.py
	```
* In a python shell:
	```python
	from selfun import createNew
	createNew.create()
	```

When prompted: 
	* Type in the directory location.
	* Add the survey name which will be used as the name of the folder and for content files.


To change the names of the files and the column headers in tables, do the following in a python shell:
```python
import pickle

# Load infofile (survey name is "surveyname")
path = 'directory/surveyname/surveyname_FileInformation.pickle'
with open(path, "rb") as input:
    file_info  = pickle.load(input)
# file_info is an instance of a class for containing all the file locations and data structures.

# To view a docstring which has example code on how to set each of the features
file_info?

# To test the file names are set correctly and the data structures are correct/
file_info.test()

# Print out all attributes and their current values:
file_info.printValues()

# Change the values of some attributes
file_info.attribute = "value of attribute"

# Repickle the class instance
file_inf.pickleInformation()
```

Once the above steps have been taken successfully, generating a selection function from the data is much easier as you don't need to worry about file locations, they all update one another.



***
## Using Isochrones <a name="isochrones"></a>

Data for the isochrones is provided in two formats:

1. Isochrone track files
	These provide the values of absolute magnitude bands given mass of the star on each isochrone. To access the data of an isochrone:
	```python
	file_name = [directory]/evoTracks/stellarprop_parsecdefault_currentmass.dill
	iso_pickle = '/media/andy/UUI/ExternalData/SFProject/stellarprop_parsecdefault_currentmass.dill'

	with open(iso_pickle, "rb") as input:
	    pi = dill.load(input)

	interpname  = "age"+str(pi.isoage)+"mh"+str(pi.isomh)
        isochrone   = pi.isodict[interpname] # NumPy array of datapoints along the isochrone
        
        Mi = isochrone[:,2] # Initial mass
        J = isochrone[:,13]
        H = isochrone[:,14]
        K = isochrone[:,15]
	```

2. Isochrone interpolants
	The interpolants are generated from a grid of age vs metallicity vs scaled initial mass (the scaled initial mass varies between 0 and 1 along any isochrone).
	
	Example: You have a comma separated txt file with 6 columns: galactic longitude (glon), galactic latitude (glat), distance (s), age, metallicity (mh), mass ([as used here](#reformat)).
	You want to know the H-band apparent magnitude and colour of the stars (neglecting dust extinction).
	```python
	import numpy as np
	import pandas as pd

	file_path = # Enter the location of the txt file here (should end in .txt)
	array = np.loadtxt(file_path)

	dataframe = pd.DataFrame(array, columns=['glon', 'glat', 's', 'age', 'mh', 'mass'])


	from selfun import IsochroneScaling

	isoCalculator = IsochroneScaling.InstrinsictoObservable()

	# If just calculating H-absolute and colour(J-K):
	isoCalculator.LoadColMag("[directory]/evoTracks/isochrone_interpolantinstances.pickle")
	colour, Habs = isoCalculator.ColMabs(dataframe.age, dataframe.mh, dataframe.mass)
	# For calculating apparent magnitude
	colour, Happ = isoCalculator.ColMapp(dataframe.age, dataframe.mh, dataframe.mass, dataframe.s)

	# If calculating all magnitudes:
	isoCalculator.LoadColMag("[directory]/evoTracks/isochrone_interpolantinstances.pickle")
	Habs, Jabs, Kabs = isoCalculator.ColMabs(dataframe.age, dataframe.mh, dataframe.mass)
	Happ, Japp, Kapp = isoCalculator.ColMabs(dataframe.age, dataframe.mh, dataframe.mass, dataframe.s)
	```

As with all modules in the package, docstrings have been constructed for the module and all internal functions so if you're unsure about the parameters, outputs or contents of a function, class or module, running help on the object should provide more information.


***
## Calculating SF probabilities <a name="sf"></a>

Here we demonstrate how to use **selfun** to generate a selection function and use it on data. 
All examples given are using Galaxia data. Folling the steps and example files should enable you to recreate the results published in Everall & Das (in prep.).

### Run selection function <a name="runsf"></a>



### Calculate selection probabilities <a name="calcsf"></a>


### Generate selection function plots <a name="plotsf"></a>


***
## Shortcuts to generate selection function <a name="shortcuts"></a>



