from seestar import surveyInfoPickler
from seestar import createNew
import healpy as hp
import os
from shutil import copyfile


class allSkySurvey():

	def __init__():

		# Ask for a spectroscopic catalogue file name
		self.spectro_fname = raw_input("Where is the spectroscopic data file? ")
		while not os.path.exists(spectro_fname):
			self.spectro_fname = raw_input("File does not exist, please give the file path: ")

		new_file = raw_input("Are you creating a new system? [y/n]")
		while not ((new_file=='y')|(new_file=='n')):
			new_file = raw_input("Please answer with y or n: ")

		if new_file == "y": 
			# Generate new info file database
			self.file_info = self.new()
		elif new_file == "n": 
			# Find 
			self.infofile = raw_input("Please give the info-file path?")
			while not os.path.exists(infofile):
				self.infofile = raw_input("File does not exist, please give the file path: ")

			# Create instance of information class
			self.file_info = surveyInfoPickler.surveyInformation()
			# Load infofile (survey name is "SURVEY-NAME")
			self.file_info.load(self.infofile)	

			# Create directory for photometric data
			photoFolder(self.file_info.spectro_folder)

		# List of photometric data files
		self.photo_files = input("List of paths to all photometric data files (Can be just one item in the list): ")
		allfiles = False
		while not allfiles:
			for path in self.photo_files:
				if not os.path.exists(path):
					allfiles=False
					self.photo_files = input("%s does not exist, enter list of file paths: " % path)
					break
				allfiles = True

		# Information about the internal data


	def new(self):

		# Create new surveyInformation dictionary for the survey

		self.infofile = create()
		# Create instance of information class
		file_info = surveyInfoPickler.surveyInformation()
		# Load infofile (survey name is "SURVEY-NAME")
		file_info.load(self.infofile)	


		# Copy spectro file into the data location
		copyfile(spectro_fname, file_info.spectro_fname)

		# Create directory for photometric data
		photoFolder(file_info.spectro_folder)

	def photoFolder(self, spectro_folder):

		# Create photometric files folder inside the data location

		# Create directory for photometric data
		photo_folder = os.path.join(spectro_folder, 'photometric/')
		if not os.path.exists( photo_folder ):
		    os.makedirs(photo_folder)

	def photoPixelFiles(theta, phi, rng_th, rng_phi, nside):

		# Put photometric  files into pixel files

		filenum = 0
		for file in self.photo_files:
			filenum += 1

			df = pd.read_csv(file)

			df['pixel'] = labelStars(df[theta], df[phi], rng_th, rng_phi, nside)

			for pix in np.arange(hp.nside2npix(nside)):

				field_stars = df[df.pixel == pix]

				fname = os.path.join(self.file_info.photo_path, str(pix)) + '.csv'
				if filenum == 1:
					field_stars.to_csv(fname, index=False)
				else:
					field_stars.to_csv(fname, mode='a', header=False, index=False)

	def spectroPixels(theta, phi, rng_th, rng_phi, nside):

		# Add pixel column to spectroscopic catalogue

		df = pd.read_csv(self.file_info.spectro_path)

		df['pixel'] = labelStars(df[theta], df[phi], rng_th, rng_phi, nside)

def labelStars(theta, phi, rng_th, rng_phi, nside):
    
    # Need to correct the range: 0<th<pi, -pi<phi<pi
    theta1 = 0 + ( (theta - rng_th[0]) * (np.pi)/(rng_th[1]-rng_th[0]) )
    phi1 = -np.pi + ( (phi - rng_phi[0]) * (2*np.pi)/(rng_phi[1]-rng_phi[0]) )
    
    pixel = hp.ang2pix(nside, theta1, phi1)
    
    return pixel
