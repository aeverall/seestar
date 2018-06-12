import os
from seestar import surveyInfoPickler


def create():

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

	# Create the directory
	os.makedirs( os.path.join(directory, folder) )

	# Initialise class
	FileInfo = surveyInfoPickler.surveyInformation()

	# Location where spectrograph survey information is stored
	FileInfo.data_path = directory

	# Folder in .data_path which contains the file informatino
	FileInfo.spectro = folder

	# Filename of spectrograph star information
	FileInfo.spectro_fname = folder + '_survey.csv'
	# Column headers for spectrograph information
	# [ fieldID, Phi, Th, magA, magB, magC]
	# magA-magB = Colour, magC = m (for selection limits)
	FileInfo.spectro_coords = ['fieldid', 'glon', 'glat', 'magA', 'magB', 'magC']

	# Filename (in FileInfo.spectro file) for field pointings
	FileInfo.field_fname = folder + '_fieldinfo.csv'
	# Column headers in field pointings
	FileInfo.field_coords = (['fieldID', 'glon', 'glat', 'Magmin', 'Magmax', 'Colmin', 'Colmax'], 'Galactic')
	# Solid angle area of fiels in deg^2
	FileInfo.field_SA = 1.
	# Data type for field IDs
	FileInfo.fieldlabel_type = str

	# Location where photometric datafiles are stored (require large storage space)
	FileInfo.photo_path = os.path.join(directory, folder)
	# Column headers in photometric data files
	FileInfo.photo_coords = ['glon', 'glat', 'magA', 'magB', 'magC']
	# File types for photometric data
	FileInfo.photo_tag = '.csv'

	# pickled file locations which will store the selection function information
	FileInfo.sf_pickle_fname = folder + '_SF.pickle'
	FileInfo.obsSF_pickle_fname = folder + '_obsSF.pickle'

	# File containing isochrone data
	FileInfo.iso_pickle_file = "evoTracks/isochrones.pickle" 
	# File location for storing information on area overlap of individual fields
	FileInfo.overlap_fname = folder + '_fieldoverlapdatabase'

	# Run the __call__ routine to setup the file locations
	FileInfo()


	# Location of pickle file which the file information will be stored in
	pklfile = os.path.join(directory, folder, folder+"_fileinfo.pickle")
	# Pickle the file information
	FileInfo.save(pklfile)

	return pklfile


if __name__ == "__main__":
	create()
