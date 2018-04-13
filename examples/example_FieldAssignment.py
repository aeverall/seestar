from selfun import FieldAssignment


assign = FieldAssignment.FieldAssignment(allsky_directory="[FILE LOCATION]/", # Location of directory containing photometric data files
										allsky_files=[...], # List of photometric files in folder (often divided up due to memory constraints)
										fieldpoint_directory = "[FILE LOCATION]/", # Location of directory in which to save field pointings files 
										pointings_path = "[FILE LOCATION]/field_file.csv", # Field pointings file location
										file_headers = [...]) # Headers of files in pointings file

assign.allsky_headers = [...] # Column headers to take from photometric catalogue
assign.allsky_angles = ['glon', 'glat'] # Column headers for longitude and latitude
assign.notnull = ['glon', 'glat'] # Columns which are not permitted to have null values
assign.allsky_total = 66853156

assign.ClearFiles() # Clear field pointings files

assign.RunAssignmentAPTPM(N_import=1000000, # Number of stars to import at one time (due to memory constraints)
			  			N_iterate=20000) # Number of stars to iterate on each iteration