'''
surveyInfoPickler.py

Classes
---------
	surveyInformation - Creating a pickled instance with all of the file locations and 
						data structure information for running the selection function.

'''

import os
import pickle
import re
import pandas as pd

class surveyInformation():

	'''
	surveyInformation - Creating a pickled instance with all of the file locations and 
						data structure information for running the selection function.

	Functions
	---------
		testFiles - Runs a test on all entries into the class to check if the file names
					have all been given correctly.

		pickleInformation - Pickles an instance of the class so that it is saved for use
							in the selection function

	Variables Required
	------------------
		data_path - str - Path to folder containing all data - e.g. '../SFdata'
		photo_path - str - Path to folder containing photometric survey data - e.g. '/media/.../2MASS'
				(This is usually kept externally due to the large file size 40+Gb)

		spectro - str - Name of spectroscopic survey folder being analysed - e.g. '/RAVE'
		spectro_fname - str - Name of file containing spectroscopic survey data - e.g. '/RAVE_wPlateIDs.csv'

		field_fname - str - File name of data file for field pointings - e.g. '/RAVE_FIELDS_wFieldIDs.csv

		spectro_coords - list of str - List of column headers taken from spectroscopic survey - 
					- e.g. ['FieldName', 'RA_TGAS', 'DE_TGAS', 'Jmag_2MASS', 'Kmag_2MASS', 'Hmag_2MASS']
		field_coords - tuple of list and str - List of column headers taken from pointings file and coord system (Equatorial or Galactic)
					- e.g. (['RAVE_FIELD', 'RAdeg', 'DEdeg'], 'Equatorial')
		photo_coords - list of str - List of column headers taken from photometric survey
					- e.g. ['RA', 'Dec', 'j', 'h', 'k']

		field_SA - float - Solid angle of all fields in the survey (this will need to be changed if SA is variable) - e.g. 28.3

		photo_pickle_fname - str - File name for pickled instance of photometric interpolants - e.g. '/2massdata_RAVEfields.pickle'
		spectro_pickle_fname - str - File name for pickled instance of spectroscopic interpolants - e.g. '/2massdata_RAVEfields.pickle'
		sf_pickle_fname - str - File name for pickled instance of physical selection function interpolants - e.g. '/2massdata_RAVEfields.pickle'

		photo_tag - str - File type tagged on the end of photometric data folders - e.g. '.csv'
		fieldlabel_type - obj - Data type used for names of fields in the pointings catalogue - e.g. str

		iso_pickle_file - str - File name for pickled isochrones instances - e.g. "/evoTracks/isochrone_distributions_resampled.pickle" 

	Example:
	--------
		Demo = surveyInformation()

		# Location where spectrograph survey information is stored
		Demo.data_path = '../../SFgithub/SFdata'

		# Folder in .data_path which contains the file informatino
		Demo.spectro = '/SpectroName'

		# Filename of spectrograph star information
		Demo.spectro_fname = '/Spectrograph_survey.csv'
		# Column headers for spectrograph information
		# [ fieldID, Phi, Th, magA, magB, magC]
		# magA-magB = Colour, magC = m (for selection limits)
		Demo.spectro_coords = ['fieldid', 'glon', 'glat', 'J', 'K', 'H']

		# Filename (in Demo.spectro file) for field pointings
		Demo.field_fname = '/Spectrograph_fieldinfo.csv'
		# Column headers in field pointings
		Demo.field_coords = (['fieldID', 'glon', 'glat', 'hmin', 'hmax', 'cmin', 'cmax'], 'Galactic')
		# Solid angle area of fiels in deg^2
		Demo.field_SA = 12.565
		# Data type for field IDs
		Demo.fieldlabel_type = np.float64

		# Location where photometric datafiles are stored (require large storage space)
		Demo.photo_path = '/photometricstorage/'
		# Column headers in photometric data files
		Demo.photo_coords = ['glon', 'glat', 'J', 'K', 'H']
		# File types for photometric data
		Demo.photo_tag = '.csv'

		# pickled file locations which will store the selection function information
		Demo.spectro_pickle_fname = '/Spectrograph_survey.pickle'
		Demo.photo_pickle_fname = '/Spectrograph_full.pickle'
		Demo.sf_pickle_fname = '/Spectrograph_SF.pickle'
		Demo.obsSF_pickle_fname = '/Spectrograph_obsSF.pickle'

		# File containing isochrone data
		Demo.iso_pickle_file = "/evoTracks/isochrones.pickle" 
		# File location for storing information on area overlap of individual fields
		Demo.overlap_fname = '/Spectrograph_fieldoverlapdatabase'

		# Run the __call__ routine to setup the file locations
		Demo()
		# testFiles checks whether the information given is accurate
		# If this is the first time running, pickle files and overlap_fname shouldn't exist
		Demo.testFiles()

		# Location of pickle file which the file information will be stored in
		pklfile = "../../SFgithub/SFdata/SpectroName/SpectrographFileInformation.pickle"
		# Pickle the file information
		Demo.pickleInformation(pklfile)
		'''

	def __init__(self):

		self.data_path = ''
		self.photo_path = ''

		self.spectro = ''
		self.spectro_fname = ''
		self.spectro_folder = os.path.join(self.data_path, self.spectro)
		self.spectro_path = os.path.join(self.spectro_folder, self.spectro_fname)

		self.field_fname = ''
		self.field_path = os.path.join(self.spectro_folder, self.field_fname)

		self.spectro_coords = None
		self.field_coords = None
		self.photo_coords = None

		self.field_SA = 0.0

		self.sf_pickle_fname = ''
		self.sf_pickle_path = os.path.join(self.spectro_folder, self.sf_pickle_fname)
		self.obsSF_pickle_fname = ''
		self.obsSF_pickle_path = os.path.join(self.spectro_folder, self.obsSF_pickle_fname)

		self.photo_tag = None
		self.fieldlabel_type = None # str 

		self.iso_pickle_file = ''
		self.iso_pickle_path = os.path.join(self.data_path, self.iso_pickle_file)
		self.iso_interp_file = ''
		self.iso_interp_path = os.path.join(self.data_path, self.iso_interp_file)

		self.overlap_fname = ''

	def __call__(self):

		self.spectro_folder = os.path.join(self.data_path, self.spectro)
		self.spectro_path = os.path.join(self.spectro_folder, self.spectro_fname)

		self.field_path = os.path.join(self.spectro_folder, self.field_fname)

		self.sf_pickle_path = os.path.join(self.spectro_folder, self.sf_pickle_fname)
		self.obsSF_pickle_path = os.path.join(self.spectro_folder, self.obsSF_pickle_fname)
		self.overlap_path = os.path.join(self.spectro_folder, self.overlap_fname)

		self.iso_pickle_path = os.path.join(self.data_path, self.iso_pickle_file)
		self.iso_interp_path = os.path.join(self.data_path, self.iso_interp_file)

		self.example_string = \
"""
# Get file names and coordinates from pickled file
pickleFile = '{directory}/{label}/{label}_FileInformation.pickle'
with open(pickleFile, "rb") as input:
    {label}  = pickle.load(input)

# Location where spectrograph survey information is stored
{label}.data_path = '../../SFgithub/SFdata'

# Folder in .data_path which contains the file informatino
{label}.spectro = '{label}'

# Filename of spectrograph star information
{label}.spectro_fname = '{label}_survey.csv'
# Column headers for spectrograph information
# [ fieldID, Phi, Th, magA, magB, magC]
# magA-magB = Colour, magC = m (for selection limits)
{label}.spectro_coords = ['fieldid', 'glon', 'glat', 'J', 'K', 'H']

# Filename (in {label}.spectro file) for field pointings
{label}.field_fname = '{label}_fieldinfo.csv'
# Column headers in field pointings
{label}.field_coords = (['fieldID', 'glon', 'glat', 'hmin', 'hmax', 'cmin', 'cmax'], 'Galactic')
# Solid angle area of fiels in deg^2
{label}.field_SA = 1.
# Data type for field IDs
{label}.fieldlabel_type = str

# Location where photometric datafiles are stored (require large storage space)
{label}.photo_path = '{label}'
# Column headers in photometric data files
{label}.photo_coords = ['glon', 'glat', 'J', 'K', 'H']
# File types for photometric data
{label}.photo_tag = '.csv'

# pickled file locations which will store the selection function information
{label}.sf_pickle_fname = '/Spectrograph_SF.pickle'
{label}.obsSF_pickle_fname = '/Spectrograph_obsSF.pickle'

# File containing isochrone data
{label}.iso_pickle_file = "evoTracks/isochrones.pickle" 
# File location for storing information on area overlap of individual fields
{label}.overlap_fname = '{label}_fieldoverlapdatabase'

# Run the __call__ routine to setup the file locations
{label}()
# testFiles checks whether the information given is accurate
# If this is the first time running, pickle files and overlap_fname shouldn't exist
{label}.testFiles()

# Pickle the file information
{label}.pickleInformation(pklfile)
""".format(label=self.spectro, directory=self.data_path)

		# Set the class instance doc string as a coding example
		self.__doc__ = self.example_string

	def pickleInformation(self, filename):

		# Can you pickle a class from inside the class?
		with open(filename, 'wb') as handle:
			pickle.dump(self, handle)

	def save(self, filename):

		# Convert attributes to dictionary
		attr_dict = vars(self)

		# Dump pickled dictionary of attributes
		with open(filename, 'wb') as handle:
			pickle.dump(attr_dict, handle)

	def load(self, filename):

		# Load pickled dictionary of attributes
        with open(filename, "rb") as input:
            file_dict  = pickle.load(input) 

        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])

	def testFiles(self):

		# Try to open folders and check file names and data structures

		# 1) data_path, spectro_path, spectro_coords
		# Use .xxx in file path to determine file type
		try:
			re_dotfile = re.compile('.+\.(?P<filetype>[a-z]+)')
			filetype = re_dotfile.match(self.spectro_path).group('filetype')
			# Import data as a pandas DataFrame
			test_data1 = getattr(pd, 'read_'+filetype)(self.spectro_path, usecols = self.spectro_coords, nrows = 5)
		except AttributeError:
			print("\nspectro_path has not been given a file name with .type on the end.")
		except IOError:
			print("\nNo file: " + self.spectro_path)

		# 2) photo_path
		if not os.path.exists(self.photo_path):
			print("\nThe path to your photometric survey, photo_path, does not exist: %s" % self.photo_path)

		# 3) field_path, field_coords, limlabels
		try:
			# Use .xxx in file path to determine file type
			re_dotfile = re.compile('.+\.(?P<filetype>[a-z]+)')
			filetype = re_dotfile.match(self.field_path).group('filetype')
			# Import data as a pandas DataFrame
			test_data2 = getattr(pd, 'read_'+filetype)(self.field_path, nrows = 5)
			print("\nYour angular coordinates will be treated as %s." % self.field_coords[1])

			for i in range(len(self.field_coords[0])):
				try: cut = test_data2[self.field_coords[0][i]].iloc[0]
				except KeyError: print("\nfield_coords column header, %s, is not in dataframe" % self.field_coords[0][i])
		except AttributeError:
			print("\nfield_path has not been given a file name with .type on the end.")
		except IOError:
			print("\nThe path to your field coordinates, field_path, does not exist: %s" % self.field_path)

		# 4) photo_pickle_path, spectro_pickle_path, sf_pickle_path, obsSF_pickle_path, overlap_path
		if not os.path.exists(self.sf_pickle_path):
			print("\nThe path to your selection function pickled instance, sf_pickle_path, does not exist: %s" % self.sf_pickle_path)
		if not os.path.exists(self.obsSF_pickle_path):
			print("\nThe path to your selection function pickled instance, obsSF_pickle_path, does not exist: %s" % self.obsSF_pickle_path)
		if not os.path.exists(self.overlap_path):
			print("\nThe path to your selection function pickled instance, obsSF_pickle_path, does not exist: %s" % self.overlap_path)

		# 5) field_SA
		print("\n The solid angle extent of your fields is %s\n" % str(self.field_SA))

		# 6) fieldlabel_type
		try:
			ftype = type(test_data2[self.field_coords[0][0]].iloc[0])
			if ftype == self.fieldlabel_type:
				print("\nYour field labels are of type %s\n" % str(ftype))
			else:
				print("\nYou stated %s for field label type but the type appears to be %s." % (str(self.fieldlabel_type), str(ftype)))
		except UnboundLocalError:
			print("\nThe tests of field_path and field_coords have not been passed so fieldlabel_type cannot be tested.")

		# 7) iso_pickle_path
		if not os.path.exists(self.iso_pickle_path):
			print("\nThe path to your isochrone pickled instance, iso_pickle_path, does not exist: %s" % self.iso_pickle_path)       	

	def printValues(self):

		print("""Location where spectrograph survey information is stored""")
		print("\ndata_path: " + self.data_path)
		print("""Location where photometric datafiles are stored (require large storage space)""")
		print("\nphoto_path: " + self.photo_path)

		print("""Folder in .data_path which contains the file informatino""")
		print("\nspectro: " + self.spectro)
		print("""Filename of spectrograph star information""")
		print("\nspectro_fname: " + self.spectro_fname)
		print("\nspectro_folder: " + self.spectro_folder)
		print("\nspectro_path: " + self.spectro_path)

		print("""Filename (in Demo.spectro file) for field pointings""")
		print("\nfield_fname: " + self.field_fname)
		print("\nfield_path: " + self.field_path)

		print("""Column headers for spectrograph information
				[ fieldID, Phi, Th, magA, magB, magC]
				magA-magB = Colour, magC = m (for selection limits)""")
		print("\nspectro_coords: " + str(self.spectro_coords))
		print("""Column headers in field pointings""")
		print("\nfield_coords: " + str(self.field_coords))
		print("""Column headers in photometric data files""")
		print("\nphoto_coords: " + str(self.photo_coords))

		print("""Solid angle area of fiels in deg^2""")
		print("\nfield_SA: " + str(self.field_SA))

		print("""pickled file locations which will store the selection function information""")
		print("\nsf_pickle_fname: " + self.sf_pickle_fname)
		print("\nsf_pickle_path: " + self.sf_pickle_path)
		print("\nobsSF_pickle_path: " + self.obsSF_pickle_path)

		print("""File types for photometric data""")
		print("\nphoto_tag: " + str(self.photo_tag))
		print("""Data type for field IDs""")
		print("\nfieldlabel_type: " + str(self.fieldlabel_type))

		print("""File containing isochrone data""")
		print("iso_pickle_file: " + self.iso_pickle_file)
		print("""File location for storing information on area overlap of individual fields""")
		print("iso_pickle_path: " + self.iso_pickle_path)

	def pythonCodeExample(self):

		print(self.example_string)