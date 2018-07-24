'''
createNew - Generates a filestructure and template files for a new selection function.

Functions
---------
create - Generates the new file structure. 


Requirements
------------

surveyInfoPickler.py
'''

import os
import numpy as np
import pandas as pd
from seestar import surveyInfoPickler

def create():

	'''
	create - Generates the new file structure. 

	Parameters
	----------
		No input parameters
			- Requests inputs from the user as the code is run.

	Returns
	-------
		pklfile: str
			- Path to the pickled survey information file.
	'''

	# Ask for a directory name
	directory = raw_input("Where is the directory? ")
	while not os.path.exists(directory):
		directory = raw_input("Directory does not exist, please give a real location: ")

	good_folder = False
	while (not good_folder):
		folder = raw_input("What survey? (will be used to label the folder and file contents) ")
		
		if os.path.exists( os.path.join(directory, folder) ):
			print("\nFolder, %s, already exists. Either delete it and try again or enter a different survey name" %\
				 os.path.join(directory, folder))
		else: good_folder = True

	good_ans = False
	while (not good_ans):
		survey_style = raw_input("Style of survey? a = multi-fibre fields, b = all-sky: ")
		if survey_style in ('a','b'):
			good_ans = True
			if survey_style == 'a': style='mf'
			else:  style='as'

	# Create the directory
	os.makedirs( os.path.join(directory, folder) )

	# Initialise class
	FileInfo = surveyInfoPickler.surveyInformation()

	# Location where spectrograph survey information is stored
	FileInfo.data_path = directory

	# Type of survey, multifibre vs allsky
	FileInfo.style = style

	# Folder in .data_path which contains the file informatino
	FileInfo.survey = folder

	# Data type for field IDs
	if style=='mf': FileInfo.fieldlabel_type = str

	# Coordinate system, Equatorial or Galactic
	FileInfo.coord_system = 'Galactic'

	# Filename of spectrograph star information
	FileInfo.spectro_fname = folder + '_survey.csv'
	# magA-magB = Colour, magC = m (for selection limits)
	FileInfo.spectro_coords = ['fieldID', 'glon', 'glat', 'Japp', 'Kapp', 'Happ']
	if style=='mf': FileInfo.spectro_dtypes = [FileInfo.fieldlabel_type, float, float, float, float, float]
	elif style=='as': FileInfo.spectro_dtypes = [None, float, float, float, float, float]
	else: print("Houston, we have a problem! code 1")

	# Filename (in FileInfo.spectro file) for field pointings
	FileInfo.field_fname = folder + '_fieldinfo.csv'
	# Column headers in field pointings and their datatypes
	if style=='mf':
		FileInfo.field_coords = ['fieldID', 'glon', 'glat', 'halfangle', 'Magmin', 'Magmax', 'Colmin', 'Colmax']
		FileInfo.field_dtypes = [FileInfo.fieldlabel_type, float, float, float, float, float, float, float]
	elif style=='as':
		FileInfo.field_coords = ['fieldID', 'Magmin', 'Magmax', 'Colmin', 'Colmax']
		FileInfo.field_dtypes = [FileInfo.fieldlabel_type, float, float, float, float]		
	else: print("Houston, we have a problem! code 2")

	# Location where photometric datafiles are stored (require large storage space)
	FileInfo.photo_path = os.path.join(directory, folder, 'photometric')
	# Column headers in photometric data files and their datatypes
	FileInfo.photo_coords = ['glon', 'glat', 'Japp', 'Kapp', 'Happ']
	FileInfo.photo_dtypes = [float, float, float, float, float]
	# File types for photometric data
	FileInfo.photo_tag = '.csv'

	FileInfo.theta_rng = (-np.pi/2, np.pi/2)
	FileInfo.phi_rng = (0, 2*np.pi)

	# pickled file locations which will store the selection function information
	FileInfo.sf_pickle_fname = folder + '_SF.pickle'
	FileInfo.obsSF_pickle_fname = folder + '_obsSF.pickle'

	# Folder and files containing isochrone data
	FileInfo.iso_folder = "isochrones"
	FileInfo.iso_data_file = "iso_fulldata.pickle" 
	FileInfo.iso_interp_file = "isochrone_interpolantinstances.pickle"

	# Default Gaussian Mixture model with 1 or 2 components
	FileInfo.spectro_model = ('GMM', 1)
	FileInfo.photo_model = ('GMM', 2)	

	# Run the __call__ routine to setup the file locations
	FileInfo()

	# Generate spectroscopic catalogue file
	createCSV(FileInfo.spectro_path, FileInfo.spectro_coords, FileInfo.spectro_dtypes)
	# Generate field information file
	createCSV(FileInfo.field_path, FileInfo.field_coords, FileInfo.field_dtypes)
	# Generate photometric catalogue file
	createPhoto(FileInfo.photo_path, FileInfo.photo_coords, FileInfo.photo_dtypes)
	# Generate isochrone folder
	os.makedirs(os.path.join(FileInfo.survey_folder, FileInfo.iso_folder))

	# Location of pickle file which the file information will be stored in
	pklfile = os.path.join(directory, folder, folder+"_fileinfo.pickle")
	FileInfo.fileinfo_path = pklfile
	# Pickle the file information
	FileInfo.saveas(pklfile)

	message = """
The files for the project have been generated.
They are located here: {folder}
Photometric files are in the subfolder: {photo_path}
Example csv files have been generated for you with the correct column headings.
""".format(folder=FileInfo.survey_folder, photo_path=FileInfo.photo_path)
	print(message)

	return pklfile


def createCSV(filelocation, headers, dtypes):

	'''
	createCSV - Producest a csv file with the required file structure

	Parameters
	----------
		filelocation: str
			- Path to the location where the file is going to be saved

		headers: list of str
			- Column headers for the table stored in the csv

		dtypes: list of type
			- Datatypes for each of the columns in the table.
	'''

	# create array with data
	data = np.zeros((len(headers), 5))

	# transform to dataframe
	df = pd.DataFrame(data.T, columns=headers)

	# Correct data types
	for i in range(len(headers)):
		df[headers[i]] = df[headers[i]].astype(dtypes[i])

	# Save csv file
	df.to_csv(filelocation, index=False, header=True)

def createPhoto(folderlocation, headers, dtypes):

	'''
	createPhoto - Produces a photometric data folder and template file.

	Parameters
	----------
		folderlocation: str
			- Path to the directory where the folder will be stored

		headers: list of str
			- Column headers for the table stored in the csv

		dtypes: list of type
			- Datatypes for each of the columns in the table.
	'''

	# Create photometric directory
	os.makedirs( folderlocation )

	# create a filename
	photo_file = os.path.join(folderlocation, 'field1.csv')

	# create csv file
	createCSV(photo_file, headers, dtypes)


if __name__ == "__main__":
	create()
