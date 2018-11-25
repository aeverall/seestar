# seestar

**seestar** is a Python package for creating and using selection functions for spectroscopic stellar surveys.

The full theory and design of the selection function can be found in Everall & Das (in prep.). Please cite this paper when using the repository.

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
## Use premade selection function <a name="PreSF"></a>

### Download data files

The files required to run **seestar** are too large to store on GitHub so they are kept separately.
Data we provide can be found [here](https://drive.google.com/drive/folders/1mz09FRP6hJPo1zPBJHP1T0BNhtDOkdGs?usp=sharing).

Each gzip file contains the selection function data for the specified survey and data release (e.g. APOGEE14 refers to DR14 of the Apache Point Observatory Galaxy Evolution Experiment).
Download the data then extract the file to the location where you wish to store the data (we recommend storing all downloaded files in the same directory).

The information held in fileinfo.pickle needs to be updated with the file locations, and some other bits of information in order to calculate the selection function.
```python
from seestar import surveyInfoPickler
path = 'PATH/TO/DIRECTORY/SURVEY-NAME/SURVEY-NAME_fileinfo.pickle'
fileinfo = surveyInfoPickler.surveyInformation(path)
```
This will raise a query:
```File location has changed, reset the file locations? (y/n)```
Respond ```y``` to change the stored fileinfo so that the correct paths to the data are saved.

Each survey folder in the database contains the following files:
* Spectrograph catalogue including crossmatch magnitudes with photometric survey. (```SURVEY_survey.csv```)
* Photometric survey data for each field in the spectrograph catalogue. (```photometric/FIELD-ID.csv```)
* Pickle files for the selection function in each coordinate system. (```SURVEY_obsSF.pickle```, ```SURVEY_SF.pickle```)
* Pickle "fieldInfo" file which stores the information on all other files for each survey. (```SURVEY_fileinfo.pickle```)
* Spectrograph field pointing locations and IDs. (```SURVEY_fieldinfo.csv```)
* Information on overlap between survey fields. (```SURVEY_fieldoverlapdatabase```)


### Calculating SF probability

All examples given are using Galaxia data (labelled as Galaxia3 as we tested with 3 fields). Following the steps and example files should enable you to recreate the results published in Everall & Das (in prep.).

To initialise prebuilt selection function:
```python
from seestar import SelectionGrid

# To initialise the prebuilt selection function:
Galaxia_sf = SeletionGrid.SFGenerator('/home/USER/PATH/Galaxia3/Galaxia3_fileinfo.pickle')
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

2. You have a comma separated txt file with six columns: galactic longitude (glon), galactic latitude (glat), apparent H magnitude (Happ), J-K colour (colour).
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

To create a new selection function, the user needs to create a new survey folder and record information on the file names and locations. Then input files containing information on the spectroscopic catalogue, the field pointings, and photometric information for each field pointing need to be created in the new folder. Preparation of the survey folder is described in more detail in PrepareSurveyFolder. Then the observed selection function can be calculated (ObsSF), as a function of colour and magnitude (any specified colour and magnitude), and the intrinsic selection function can be calculated (IntSF), as a function of distance, age, metallicity, and mass, although this assumes that the observed selection function has been calculated as a function of H-band apparent magnitude and J-K colour. PlotSF describes how to construct plots of the selection function.

### Preparing the survey folder

A new folder in the data directory needs to be created for the survey by doing the following in a python shell:
```python
from seestar import createNew
createNew.create()
```

This will request some inputs:
```
Where is the directory? PATH/TO/DIRECTORY
What survey? (will be used to label the folder and file contents) SURVEY
Style of survey? a = multi-fibre fields, b = all-sky: a/b
```
The ```PATH/TO/DIRECTORY``` is the directory where you wish to store folders for each survey.
The SURVEY is the label you wish to give to the survey (e.g. APOGEE14).
If a folder with ```SURVEY-NAME``` exists in ```PATH/TO/DIRECTORY/```, you will have to provide a different name.
Surveys, such as APOGEE and RAVE, are multi-fibre spectrographs for which field pointings provide a well defined set of fields.
Gaia is an all-sky survey as it systematically scans the entire sky without predefined fields.

A folder labeled SURVEY-NAME will be generated in the location PATH/TO/DIRECTORY and will contain the following:
* ```SURVEY_fileinfo.pickle``` - pickled dictionary of survey information (file locations and data structures).
* ```SURVEY_survey.csv``` - spectroscopic catalogue template.
* ```SURVEY_fieldinfo.csv``` - spectroscopic field pointing catalogue template.
* ```photometric/field1.csv``` - folder for photometric catalogue files for each field in the spectroscopic survey (an example template file, field1.csv, is also included).
* ```isochrones/``` - folder for isochrone files (You'll need to move these in yourself)

Once you have created this folder, you must replace the template files with real field files:
* Spectroscopic catalogue (```SURVEY_survey.csv```). This file will be a comma separated file with at least the following five columns (appropriately labelled): galactic longitude in radians ('glon'), galactic latitude in radians ('glat'), apparent magnitudes ('Happ', 'Japp', 'Kapp'), field id tag for the star ('fieldID'). The file can have other columns too but they won't be used.
* Photometric catalogue for each field in the spectroscopic catalogue (```photometric/FIELD-ID.csv```). Comma separated file listing all stars on the field pointing in the photometric catalogue. It will have at least the following four columns (appropriately labelled): galactic longitude in radians ('glon'), galactic latitude in radians ('glat'), apparent magnitudes ('Happ', 'Japp', 'Kapp'). The file can have other columns too but they won't be used.
* Locations and IDs of the spectroscopic field pointings (```SURVEY_fieldinfo.csv```). This file gives the central galactic longitude in radians ('glon') and galactic latitude in radians ('glat') of each field, half angle in radians ('halfangle') and the color and magnitude limits imposed by the spectroscopic survey ('Magmin', 'Magmax', 'Colmin', 'Colmax'). If none are imposed, write "NoLimit".
* PARSEC isochrone files can taken from [here](https://drive.google.com/drive/folders/1YOZyHzdMP5-wgDVv-SlDXEVGcWagGG3-?usp=sharing). Move isochrone_interpolantinstances.pickle into the ```isochrones/``` folder. Without the isochrones, you can still generate the selection function in observable coordinates.


Use the fileinfo file to test whether the data files are all correctly formatted and in the right locations.
```python
from seestar import surveyInfoPickler

# Create instance of information class
path = 'PATH/TO/DIRECTORY/SURVEY-NAME/SURVEY-NAME_fileinfo.pickle'
fileinfo = surveyInfoPickler.surveyInformation(path)
# fileinfo is an instance of a class for containing all the file locations and data structures.

# Run a test of the structure of files in the selection function directory to check if they're ready.
fileinfo.testFiles()

# Repickle the class instance
fileinfo.save()
```

### Field Assignment

As mentioned previously, the folder, ```photometric/```, in the survey directory stores the photometric survey data as a file for every field in the survey with the photometric stars which are on that field. You can generate these stars yourself, or use the given field assignment code:

```python
from seestar import FieldAssignment
# Path to fileinfo.pickle file generated when running createNew
fileinfo_path = '/home/PATH/TO/DIRECTORY/SURVEY/SURVEY_fileinfo.pickle'
# List of paths to all photometric catalogue files (photometric catalogues can be large so often multiple files are used)
files = ["/home/path/to/photo_cat/photofile1", "/home/path/to/photo_cat/photofile2", "/home/path/to/photo_cat/photofile3"]
# Run the field assignment
FA = FieldAssignment.FieldAssignment(fileinfo_path, files)
```
This may raise some warnings if the datastructure in the files is not the same as suggested in fileinfo.
Respond n to stop running and fix the warnings (such as changing column headers and coordinates to radians).
Respond y to ignore the warnings and continue.

The size of iteration steps is decided by checking the memory available on your system.
(Hasn't been tested on a cluster so not sure how well it works there.)

### Generating the selection function in observable coordinates (as a function of colour and magnitude)

Generate the selection function.
```python
from seestar import SelectionGrid

Survey_sf = SeletionGrid.SFGenerator('/home/PATH/TO/DIRECTORY/SURVEY/SURVEY_fileinfo.pickle')
```
This will raise the question:
```
Would you like the selection function in: a) observable, b) intrinsic, c) both? (return a, b or c)
```
* ```a``` - Generates a selection function in observable coordinates (magnitude, colour). Doesn't require Isochrone data if this is unavailable.
* ```b``` - Generates a selection function in intrinsic coordinates (age, metallicity, mass, distance). Requires the isochrone files.
* ```c``` - Generates both of the above and keeps them loaded in to be used when wanted.

If the selection function has been previously run, the following will automatically appear:
```
Path to intrinsic SF (Galaxia3_new_SF.pickle) exists. Load SF in from here? (y/n)
Path to observable SF (Galaxia3_new_obsSF.pickle) exists. Use this to ? (y/n)
```
* y - Loads in the previously generated selection function (done in seconds)
* n - Generates a new selection function which will overwrite the previous ones.
(Calculating the observable SF from scratch can take hours depending on the survey)


Having created the selection function instance for ```SURVEY-NAME```, we now wish to calculate selection probabilities of stars:

1. You have a comma separated txt file with six columns: galactic longitude (glon), galactic latitude (glat), distance (s), age, metallicity (mh), mass.
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
dataframe = Galaxia_sf(dataframe, method='intrinsic',
				coords=['age', 'mh', 's', 'mass'], angle_coords=['glon', 'glat'])
# method='intrinsic' means calculating the selection function using intrinsic properties (i.e. age, metallicity, distance and mass).

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

2. Isochrone data file
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
