import pickle, os
import surveyInfoPickler

def replaceNames(directory):

	directories = []

	folders = [x for x in os.listdir(directory) if os.path.isdir( os.path.join(directory, x) )]

	# Scan through datafolders in directory
	for folder in folders:
		# Scan through files in each folder
		for file in os.listdir( os.path.join(directory, folder) ):
			# Search for info files
			if file.endswith('Information.pickle'):

				# Load infofile
				with open(os.path.join(directory, folder, file), "rb") as input:
					file_info  = pickle.load(input)

				# Replace directory with the correct one
				file_info.data_path = directory
				# Replace photo_path with the directory name
				file_info.photo_path = os.path.join(directory,folder)
				# Update remaining entries
				file_info()
				
				# Repickle file
				file_info.pickleInformation( os.path.join(directory, folder, file) )

				directories.append(os.path.join(directory, folder, file))

	print("New data_path ("+directory+") set for the following files : ")
	for file in directories: print(file)


if __name__ == '__main__':

	directory = raw_input("Location of data directory: ")

	replaceNames(directory)