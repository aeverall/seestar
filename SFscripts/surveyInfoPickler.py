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
	'''

	def __init__(self):

		self.data_path = ''
		self.photo_path = ''

		self.spectro = ''
		self.spectro_fname = ''
		self.spectro_folder = self.data_path + self.spectro
		self.spectro_path = self.spectro_folder + self.spectro_fname

		self.field_fname = ''
		self.field_path = self.spectro_folder + self.field_fname

		self.spectro_coords = None
		self.field_coords = None
		self.photo_coords = None

		self.field_SA = 0.0

		self.photo_pickle_fname = '' 
		self.photo_pickle_path = self.spectro_folder + self.photo_pickle_fname
		self.spectro_pickle_fname = ''
		self.spectro_pickle_path = self.spectro_folder + self.spectro_pickle_fname
		self.sf_pickle_fname = ''
		self.sf_pickle_path = self.spectro_folder + self.sf_pickle_fname
		self.obsSF_pickle_fname = ''
		self.obsSF_pickle_path = self.spectro_folder + self.obsSF_pickle_fname

		self.photo_tag = None
		self.fieldlabel_type = None # str 

		self.iso_pickle_file = ''
		self.iso_pickle_path = self.data_path + self.iso_pickle_file

	def __call__(self):

		self.spectro_folder = self.data_path + self.spectro
		self.spectro_path = self.spectro_folder + self.spectro_fname

		self.field_path = self.spectro_folder + self.field_fname

		self.photo_pickle_path = self.spectro_folder + self.photo_pickle_fname
		self.spectro_pickle_path = self.spectro_folder + self.spectro_pickle_fname
		self.sf_pickle_path = self.spectro_folder + self.sf_pickle_fname
		self.obsSF_pickle_path = self.spectro_folder + self.obsSF_pickle_fname

		self.iso_pickle_path = self.data_path + self.iso_pickle_file

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

		# 3) field_path, field_coords
		try:
			# Use .xxx in file path to determine file type
			re_dotfile = re.compile('.+\.(?P<filetype>[a-z]+)')
			filetype = re_dotfile.match(self.field_path).group('filetype')
			# Import data as a pandas DataFrame
			test_data2 = getattr(pd, 'read_'+filetype)(self.field_path, usecols = self.field_coords[0], nrows = 5)
			print("\nYour angular coordinates will be treated as %s." % self.field_coords[1])
		except AttributeError:
			print("\nfield_path has not been given a file name with .type on the end.")
		except IOError:
			print("\nThe path to your field coordinates, field_path, does not exist: %s" % self.field_path)

		# 4) photo_pickle_path, spectro_pickle_path, sf_pickle_path
		if not os.path.exists(self.photo_pickle_path):
			print("\nThe path to your photometric pickled instance, photo_pickle_path, does not exist: %s" % self.photo_pickle_path)
		if not os.path.exists(self.spectro_pickle_path):
			print("\nThe path to your spectroscopic pickled instance, spectro_pickle_path, does not exist: %s" % self.spectro_pickle_path)
		if not os.path.exists(self.sf_pickle_path):
			print("\nThe path to your selection function pickled instance, sf_pickle_path, does not exist: %s" % self.sf_pickle_path)
		if not os.path.exists(self.obsSF_pickle_path):
			print("\nThe path to your selection function pickled instance, obsSF_pickle_path, does not exist: %s" % self.obsSF_pickle_path)

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

		print("\ndata_path: " + self.data_path)
		print("\nphoto_path: " + self.photo_path)

		print("\nspectro: " + self.spectro)
		print("\nspectro_fname: " + self.spectro_fname)
		print("\nspectro_folder: " + self.spectro_folder)
		print("\nspectro_path: " + self.spectro_path)

		print("\nfield_fname: " + self.field_fname)
		print("\nfield_path: " + self.field_path)

		print("\nspectro_coords: " + str(self.spectro_coords))
		print("\nfield_coords: " + str(self.field_coords))
		print("\nphoto_coords: " + str(self.photo_coords))

		print("\nfield_SA: " + str(self.field_SA))

		print("\nphoto_pickle_fname: " + self.photo_pickle_fname)
		print("\nphoto_pickle_path: " + self.photo_pickle_path)
		print("\nspectro_pickle_fname: " + self.spectro_pickle_fname)
		print("\nspectro_pickle_path: " + self.spectro_pickle_path)
		print("\nsf_pickle_fname: " + self.sf_pickle_fname)
		print("\nsf_pickle_path: " + self.sf_pickle_path)
		print("\nobsSF_pickle_path: " + self.obsSF_pickle_path)

		print("\nphoto_tag: " + str(self.photo_tag))
		print("\nfieldlabel_type: " + str(self.fieldlabel_type))

		print("iso_pickle_file: " + self.iso_pickle_file)
		print("iso_pickle_path: " + self.iso_pickle_path)

	def pickleInformation(self, filename):

		# Can you pickle a class from inside the class?
		with open(filename, 'wb') as handle:
			pickle.dump(self, handle)




"""
Example of the inputs to fields for RAVE and APOGEE

		        # RAVE column headers and file names:
        if survey=='RAVE':
            rave_path = data_path + "/RAVE"

            star_coords = ['FieldName', 'RA_TGAS', 'DE_TGAS', 'Jmag_2MASS', 'Kmag_2MASS', 'Hmag_2MASS']
            star_path = rave_path + '/RAVE_wPlateIDs.csv'
            field_coords = (['RAVE_FIELD', 'RAdeg', 'DEdeg'], 'Equatorial')
            field_path = rave_path + '/RAVE_FIELDS_wFieldIDs.csv'
            self.tmass_coords = ['RA', 'Dec', 'j', 'h', 'k']

            tmass_path = tmass_path + '/RAVE-2MASS_fieldbyfield_29-11/RAVEfield_'

            SA = 28.3
            tmass_pickle = rave_path + '/2massdata_RAVEfields.pickle'
            survey_pickle = rave_path + '/surveydata_RAVEfields.pickle'
            sf_pickle = rave_path + '/sf_RAVEfields.pickle'
            self.tmasstag = '.csv'
            self.fieldlabel_type = str

            if testbool:
                tmass_pickle = rave_path + '/2massdata_RAVEfields_test.pickle'
                survey_pickle = rave_path + '/surveydata_RAVEfields_test.pickle'
                sf_pickle = rave_path + '/sf_RAVEfields_test.pickle'

        # Apogee column headers and file names:
        if survey=='Apogee':
            apg_path = data_path + "/Apogee"

            star_coords = ['# location_id', 'ra', 'dec', 'j', 'k', 'h']
            field_coords = (['FieldName', 'RA', 'Dec'], 'Equatorial')
            star_path = apg_path + '/TGAS_APOGEE_supp_keplercannon_masses_ages.csv'
            field_path = apg_path + '/apogeetgasdr14_fieldinfo.csv'
            self.tmass_coords = ['RA', 'Dec', 'j', 'h', 'k']
            
            tmass_path = tmass_path + '/Apogee-2MASS/APOGEEfield_'
            self.tmasstag = '.csv'

            SA = 7
            tmass_pickle = apg_path + '/2massdata_apogeeFields_test.pickle'
            survey_pickle = apg_path + '/surveydata_apogeeFields_test.pickle'
            sf_pickle = apg_path + '/sf_apogeefields.pickle'

            self.fieldlabel_type = float
"""