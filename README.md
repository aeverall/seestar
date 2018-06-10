# seestar

**seestar** is a Python package for creating and using Selection Functions for spectroscopic stellar surveys.

The full theory and design of the selection function can be found in Everall & Das (in prep.), please cite this paper when using the repository.

The purpose of **seestar** is to provide an *easy-to-use*, *fast processing* and *mathematically consistent* method for calculating selection functions for spectroscopic stellar surveys. We hope that this will consolidate the many different methods currently used to calculate selection functions. We also provide precalculated selection functions for popular spectroscopic surveys.

This project will aim to constantly evolve and improve with developments to survey data and requirements of the research being performed. We will list improvements and updates to come as well as the selection functions we are building [here](Future.md).

In the remainder of this README file, we will explain how to install and run **seestar** to calculate selection functions for your favourite surveys.


***
# Contents:
1. [Install package](#install)
2. [Calculate selection functions](#SF)
	- [Premade selection functions](#PreSF)
	- [New selection functions](#NewSF)
3. [Isochrone Calculator](#isochrones)


***
# Download and install code package <a name="install"></a>

Go to the location where you would like to store the repository.

```
$ git clone https://github.com/aeverall/seestar.git
$ cd seestar
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



***
# Calculate selection functions <a name="SF"></a>

These sections describe how to calculate selection function probabilities, using our [premade selection functions](#PreSF) and from scratch for [new selection functions](#NewSF).
Before doing this, please follow the [install](#install) instructions to set up **seestar** on your device.


***
## Use premade selection function <a name="PreSF"></a>

### Download data files

The files required to run **seestar** are too large to store on GitHub so they are kept separately.
Data we provide can be found [here](#https://drive.google.com/drive/folders/1mz09FRP6hJPo1zPBJHP1T0BNhtDOkdGs?usp=sharing).

Each gzip file contains the selection function data for the specified survey and data release (e.g. APOGEE14 refers to DR14 of the Apache Point Observatory Galaxy Evolution Experiment).
Download the data then extract the file to the location where you wish to store the data (we recommend storing all downloaded files in the same directory).

The code repository now doesn't know where the data is stored. To give the repository the location, run the following in a Python shell:
```python
from seestar import setdatalocation
setdatalocation.replaceNames('/home/USER/PATH')
```
Use an absolute directory location (/home/USER/PATH) rather than relative locations (../USER/PATH)

Each survey folder in the database contains the following files:
* Spectrograph catalogue including crossmatch magnitudes with photometric survey. (SURVEY_survey.csv)
* Photometric survey data for each field in the spectrograph catalogue. (photometric/FIELD-ID.csv)
* Pickle files for the selection function in each coordinate system. (SURVEY_obsSF.pickle, SURVEY_SF.pickle)
* Pickle "fieldInfo" file which stores the information on all other files for each survey. (SURVEY_fileinfo.pickle)
* Spectrograph field pointing locations and IDs. (SURVEY_fieldinfo.csv)
* Information on overlap between survey fields. (SURVEY_fieldoverlapdatabase)


### Calculating SF probability

All examples given are using Galaxia data (labelled as Galaxia3 as we tested with 3 fields). Folling the steps and example files should enable you to recreate the results published in Everall & Das (in prep.).

To initialise prebuilt selection function:
```python
from seestar import SelectionGrid

# To initialise the prebuilt selection function:
Galaxia_sf = SeletionGrid.SFGenerator('home/USER/PATH/Galaxia3/Galaxia3_fileinfo.pickle')
```

Having created the selection function instance for Galaxia, we now wish to calculate selection probabilities of stars:

1. You have a comma separated txt file with 6 columns: galactic longitude (glon), galactic latitude (glat), distance (s), age, metallicity (mh), mass..
You want to know the probability of each star in the dataset being included in the survey.
```python
import numpy as np
import pandas as pd

# Load data into a numpy array
file_path = 'PATH/TO/FILE.txt'
array = np.loadtxt(file_path)

# Generate a pandas dataframe from the array
dataframe = pd.DataFrame(array, columns=['glon', 'glat', 's', 'age', 'mh', 'mass'])

# Calculation of selection function
# - the dataframe is returned with columns for the fields of the stars and selection probability.
dataframe = Galaxia_sf(dataframe, method='int',
				coords=['age', 'mh', 's', 'mass'], angle_coords=['glon', 'glat'])
# method='int' means calculating the selection function using intrinsic properties (i.e. age, metallicity, distance and mass).

dataframe.union # The column of selection function probabilities
```

2. You have a comma separated txt file with 6 columns: galactic longitude (glon), galactic latitude (glat), apparent H magnitude (Happ), J-K colour (colour).
You want to know the probability of each star in the dataset being included in the survey.
```python
import numpy as np
import pandas as pd

# Load data into a numpy array
file_path = 'PATH/TO/FILE.txt'
array = np.loadtxt(file_path)

# Generate a pandas dataframe from the array
dataframe = pd.DataFrame(array, columns=['glon', 'glat', 'Happ', 'colour'])

# Calculation of selection function 
# - the dataframe is returned with columns for the fields of the stars and selection probability.
dataframe = Galaxia_sf(dataframe, method='observable',
				coords=['Happ', 'colour'], angle_coords=['glon', 'glat'])
# Method='observable' means calculating the selection function using observable properties 
# (i.e. apparent magnitude and colour).

dataframe.union # The column of selection function probabilities
```

### Generate selection function plots.

We also include several methods for plotting out selection function probabilities in various coordinate systems.

```python
# Contour plot of selection function in distance vs age plane for metallicity=0. and integrating over the Kroupa IMF.
field = 1.0
DistributionPlots.plotSpectroscopicSF2(Galaxia_sf.instanceIMFSF, Galaxia_sf.obsSF, field, nlevels=18, mh=0.0
                                  	title=r"$\mathrm{P}(\mathrm{S}|\mathrm{[M/H] = -0.2},\, s,\, \tau)$")
```
For more information on the [Kroupa IMF](https://ui.adsabs.harvard.edu/#abs/2001MNRAS.322..231K/abstract).


***
## Create new selection function <a name="NewSF"></a>

### File formatting

The repository is designed to run csv files with well defined headers. Therefore when adding new surveys to the selection function, any files need to be converted into the correct format.

Example: You have a comma separated txt file with 6 columns: galactic longitude (glon), galactic latitude (glat), apparent magnitudes (Japp, Happ, Kapp).
```python
import numpy as np
import pandas as pd

file_path = 'PATH/TO/DATA/raw_data.txt' # Enter the location of the txt file here (should end in .txt)
array = np.loadtxt(file_path)

dataframe = pd.DataFrame(array, columns=['glon', 'glat', 'Japp', 'Kapp', 'Happ'])

new_file_path = 'PATH/TO/DATA/reformatted_data.csv' # File path for new csv file (should end in .csv)
dataframe.to_csv(new_file_path)
```
This also demonstrates how to create a pandas dataframe, the main tool in pandas. I would highly recommend playing around with the dataframes to find out how useful they are (especially if you usually use numpy arrays for handling data tables).
In the repository /examples/ex_pandas.py contains some basic examples of how to use dataframes. 

### Isochrone data

To generate the full selection function, isoPARSEC.tar.gz must be downloaded and extracted from [here](https://drive.google.com/drive/folders/1mz09FRP6hJPo1zPBJHP1T0BNhtDOkdGs?usp=sharing).


### Survey file information <a name="infofile"></a>

For each survey, a class containing the description of file locations and datatypes is required. The pickle instances are called *survey*_fileinfo.pickle for the premade surveys but for new selection functions these need to be generated.

A folder in the data directory can be created by doing one of the following:
In command line:
```
$ python seestar/createNew.py
```
In a python shell:
```python
from seestar import createNew
createNew.create()
```

This will request some inputs:
```
Where is the directory? PATH/TO/DIRECTORY
What survey? (will be used to label the folder and file contents) SURVEY-NAME
```
The PATH/TO/DIRECTORY is the directory where you wish to store folders for each survey.
The SURVEY-NAME is the label you wish to give to the survey (e.g. APOGEE14).
If a folder with SURVEY-NAME exists in PATH/TO/DIRECTORY/, you will have to provide a different name.

A folder labeled SURVEY-NAME will be generated in the location PATH/TO/DIRECTORY and will contain a SURVEY-NAME_fileinfo.pickle file.
The information held in this file will need to be changed to match the data of the survey.
```python
import pickle

# Load infofile (survey name is "surveyname")
path = 'PATH/TO/DIRECTORY/SURVEY-NAME/SURVEY-NAME_fileinfo.pickle'
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
file_info.pickleInformation()
```
Once you have pickled the SURVEY-NAME_fileinfo.pickle file, the selection function will be able to use those file locations and structures.


### Calculating SF probability

Generate the selection function.
```python
from seestar import SelectionGrid

Survey_sf = SeletionGrid.SFGenerator('PATH/TO/DIRECTORY/SURVEY-NAME/SURVEY-NAME_fileinfo.pickle', 
						ColMagSF_exists=False)
```

On completion, this automatically saves the selection function, after which it can be reloaded much faster:
```python
Survey_sf = SeletionGrid.SFGenerator('PATH/TO/DIRECTORY/SURVEY-NAME/SURVEY-NAME_fileinfo.pickle', 
						ColMagSF_exists=True)
```

Having created the selection function instance for SURVEY-NAME, we now wish to calculate selection probabilities of stars:

1. You have a comma separated txt file with 6 columns: galactic longitude (glon), galactic latitude (glat), distance (s), age, metallicity (mh), mass.
You want to know the probability of each star in the dataset being included in the survey.
```python
import numpy as np
import pandas as pd

# Load data into a numpy array
file_path = 'PATH/TO/FILE.txt'
array = np.loadtxt(file_path)

# Generate a pandas dataframe from the array
dataframe = pd.DataFrame(array, columns=['glon', 'glat', 's', 'age', 'mh', 'mass'])

# Calculation of selection function
# - the dataframe is returned with columns for the fields of the stars and selection probability.
dataframe = Galaxia_sf(dataframe, method='int',
				coords=['age', 'mh', 's', 'mass'], angle_coords=['glon', 'glat'])
# method='int' means calculating the selection function using intrinsic properties (i.e. age, metallicity, distance and mass).

dataframe.union # The column of selection function probabilities
```

2. You have a comma separated txt file with 6 columns: galactic longitude (glon), galactic latitude (glat), apparent H magnitude (Happ), J-K colour (colour).
You want to know the probability of each star in the dataset being included in the survey.
```python
import numpy as np
import pandas as pd

# Load data into a numpy array
file_path = 'PATH/TO/FILE.txt'
array = np.loadtxt(file_path)

# Generate a pandas dataframe from the array
dataframe = pd.DataFrame(array, columns=['glon', 'glat', 'Happ', 'colour'])

# Calculation of selection function 
# - the dataframe is returned with columns for the fields of the stars and selection probability.
dataframe = Galaxia_sf(dataframe, method='observable',
				coords=['Happ', 'colour'], angle_coords=['glon', 'glat'])
# Method='observable' means calculating the selection function using observable properties 
# (i.e. apparent magnitude and colour).

dataframe.union # The column of selection function probabilities
```


### Generate selection function plots.

We also include several methods for plotting out selection function probabilities in various coordinate systems.

```python
# Contour plot of selection function in distance vs age plane for metallicity=0. and integrating over the Kroupa IMF.
field = 1.0
DistributionPlots.plotSpectroscopicSF2(Galaxia_sf.instanceIMFSF, Galaxia_sf.obsSF, field, nlevels=18, mh=0.0
                                  	title=r"$\mathrm{P}(\mathrm{S}|\mathrm{[M/H] = -0.2},\, s,\, \tau)$")
```
For more information on the [Kroupa IMF](https://ui.adsabs.harvard.edu/#abs/2001MNRAS.322..231K/abstract).


***
## Isochrone Calculator <a name="isochrones"></a>

Downloaded and extracted isoPARSEC.tar.gz from [here](https://drive.google.com/drive/folders/1mz09FRP6hJPo1zPBJHP1T0BNhtDOkdGs?usp=sharing).

Data for the isochrones is provided in two formats:

1. Isochrone track files
	These provide the values of absolute magnitude bands given mass of the star on each isochrone. To access the data of an isochrone:
	```python
	import dill
	iso_pickle = 'PATH/TO/DIRECTORY/isoPARSEC/stellarprop_parsecdefault_currentmass.dill'

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
	
	file_path = 'PATH/TO/FILE.txt'
	array = np.loadtxt(file_path)
	
	dataframe = pd.DataFrame(array, columns=['glon', 'glat', 's', 'age', 'mh', 'mass'])
	
	
	from seestar import IsochroneScaling
	isoCalculator = IsochroneScaling.IntrinsicToObservable()
	
	# If just calculating H-absolute and colour(J-K):
	isoCalculator.LoadColMag("*path*/isoPARSEC/isochrone_interpolantinstances.pickle")
	colour, Habs = isoCalculator.ColourMabs(dataframe.age, dataframe.mh, dataframe.mass)
	# For calculating apparent magnitude
	colour, Happ = isoCalculator.ColourMapp(dataframe.age, dataframe.mh, dataframe.mass, dataframe.s)
	
	# If calculating all magnitudes:
	isoCalculator.LoadMagnitudes("[directory]/evoTracks/isochrone_magnitudes.pickle")
	Habs, Jabs, Kabs = isoCalculator.AbsMags(dataframe.age, dataframe.mh, dataframe.mass)
	Happ, Japp, Kapp = isoCalculator.AppMags(dataframe.age, dataframe.mh, dataframe.mass, dataframe.s)
	```

As with all modules in the package, docstrings have been constructed for the module and all internal functions so if you're unsure about the parameters, outputs or contents of a function, class or module, running help on the object should provide more information.




