# seestar


**seestar** is a Python package for creating and using selection functions for spectroscopic stellar surveys.

The full theory and design of the selection function can be found in [Everall & Das 2019](https://arxiv.org/abs/1902.10485). Please cite this paper when using the repository.
*[Data]() and an [example notebook](https://github.com/aeverall/seestar/tree/master/examples) for reproducing the Galaxia example are available. The example notebook is in the Github repositoy [examples/]() folder*

The purpose of **seestar** is to provide an *easy-to-use*, *fast processing* and *mathematically consistent* method for calculating selection functions for spectroscopic stellar surveys. We hope that this will consolidate the many different methods currently used to calculate selection functions. We also provide precalculated selection functions for popular spectroscopic surveys.

This project will aim to constantly evolve and improve with developments to survey data and requirements of the research being performed. We will list improvements and updates to come as well as the selection functions we are building [here](Future.md).

In the remainder of this README file, we will explain how to install and run **seestar** to calculate selection functions for your favourite surveys.


***
# Contents:
1. [Install package](#install)
2. [Calculate selection functions](#SF)
3. [Isochrone Calculator](#isochrones)


***
# Download and install code package <a name="install"></a>

### From Github
Go to the location where you would like to store the repository.
```
$ git clone https://github.com/aeverall/seestar.git
$ cd seestar
$ python setup.py install
```
Or, if working on a hosted server without admin privaleges:
```
$ python setup.py install --user
```
which will install the build to your ~/.local directory.

The package requires the following dependencies:
* [NumPy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
	* For those unfamiliar with pandas, it is a package structured around numpy which provides some facilities for data table manipulation which are more convenient and easier to use. Where possible instructions on how to manipulate and reformat pandas dataframes (similar to a numpy array) are given.
* [SciPy](https://www.scipy.org/)
* [re (regex for python)](https://docs.python.org/2/library/re.html)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [pickle](https://docs.python.org/2/library/pickle.html)

The code is built for Python 2.7 and is currently incompatible with Python 3.5.
(We're working on making the repository compatible with both python versions)

***
# Calculate selection functions <a name="SF"></a>

These sections describe how to calculate selection function probabilities, using our [premade selection functions](#PreSF) and from scratch for [new selection functions](#NewSF).
Before doing this, please follow the [install](#install) instructions to set up **seestar** on your device.


***
### Download data files

The files required to run **seestar** are too large to store on GitHub so they are kept separately.
Data we provide can be found [here](https://drive.google.com/drive/folders/1mz09FRP6hJPo1zPBJHP1T0BNhtDOkdGs?usp=sharing).

Each gzip file contains the selection function data for the specified survey and data release (e.g. APOGEE14 refers to DR14 of the Apache Point Observatory Galaxy Evolution Experiment).
Download the data then extract the file to the location where you wish to store the data (we recommend storing all downloaded files in the same directory).

Each survey folder in the database contains the following files:
* Spectrograph catalogue including crossmatch magnitudes with photometric survey. (```SURVEY_survey.csv```)
* Photometric survey data for each field in the spectrograph catalogue. (```photometric/FIELD-ID.csv```)
* Pickle files for the selection function in each coordinate system. (```SURVEY_obsSF.pickle```, ```SURVEY_intSF.pickle```)
* Spectrograph field pointing locations and IDs. (```SURVEY_fieldinfo.csv```)

### Calculating SF probability

All examples given are using Galaxia data. To do this yourself, you can run examples/Galaxia_selection.ipynb.
You only need to replace the *folder* string with the path to the downloaded data.
Following the steps and example files should enable you to recreate the results published in Everall & Das (in prep.).

To initialise prebuilt selection function:
```python
from seestar import SelectionGrid

# To initialise the prebuilt selection function:
Galaxia_sf = SelectionGrid.SFGenerator(get_spectro, get_photo, pointings, ncores=3,
                              				spectro_model=('GMM', 3), photo_model=('GMM', 3))
# get_spectro and get_photo are functions you design to retrieve data given a field ID.
# You can use them to call data from a file or querying servers etc.

# Load in a prebuilt selection function...
# Load in observable selection function
Galaxia_sf.load_obsSF(obsSF_path='path_to_obsSF')
# Load in intrinsic selection function
Galaxia_sf.load_intSF(intSF_path='path_to_intSF')

# Or generate a new selection function...
# Generate observable selection function
SF.gen_obsSF(folder+'/Galaxia_obsSF2.pickle')
# Generate intrinsic selection function
SF.gen_intSF(folder+'/Galaxia_intSF2.pickle', IsoCalculator)
```

IsoCalculator is a class which calculates Magnitudes given intrinsic parameters. We explain more [here](#isochrones).

Having created the selection function instance for Galaxia, we now wish to calculate selection probabilities of stars:

1. You have a comma separated txt file with columns: galactic longitude (glon), galactic latitude (glat), distance (s), age, metallicity (mh), mass..
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

2. You have a comma separated txt file with columns: galactic longitude (glon), galactic latitude (glat), apparent H magnitude (Happ), J-K colour (colour).
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

***
## Isochrone calculator <a name="isochrones"></a>

Downloaded isochrone data files from [here](https://drive.google.com/drive/folders/1YOZyHzdMP5-wgDVv-SlDXEVGcWagGG3-?usp=sharing).

Data for the isochrones is provided in two formats:

1. Isochrone interpolants

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
IsoCalculator = IsochroneScaling.IntrinsicToObservable()

# If just calculating H-absolute and colour(J-K):
IsoCalculator.LoadColMag("*path*/isoPARSEC/isochrone_interpolantinstances.pickle")
colour, Habs = IsoCalculator.ColourMabs(dataframe.age, dataframe.mh, dataframe.mass)
# For calculating apparent magnitude
colour, Happ = IsoCalculator.ColourMapp(dataframe.age, dataframe.mh, dataframe.mass, dataframe.s)

# If calculating all magnitudes:
IsoCalculator.LoadMagnitudes("[directory]/evoTracks/isochrone_magnitudes.pickle")
Habs, Jabs, Kabs = IsoCalculator.AbsMags(dataframe.age, dataframe.mh, dataframe.mass)
Happ, Japp, Kapp = IsoCalculator.AppMags(dataframe.age, dataframe.mh, dataframe.mass, dataframe.s)
```

2. Isochrone data file.

	This contains all the data on the isochrones before they've been resampled to minimize displacement errors in calculating the intrinsic selection function.
	I would advise that if the interpolantinstance.pickle file is available, use that as it's far simpler and less memory intensive.

	To load this into the Isochrone Calculator, perform the following:

```python
import numpy as np
import pandas as pd

file_path = "*path*/isoPARSEC/iso_fulldata.pickle"

# Create isochrone calculator
from seestar import IsochroneScaling
IsoCalculator = IsochroneScaling.IntrinsicToObservable()
IsoCalculator.CreateFromIsochrones(fileinfo.iso_data_path)
IsoCalculator.pickleColMag(fileinfo.iso_interp_path)
```

As with all modules in the package, docstrings have been constructed for the module and all internal functions so if you're unsure about the parameters, outputs or contents of a function, class or module, running help on the object should provide more information.
