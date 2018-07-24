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
import numpy as np

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

    Attributes
    ------------------
        data_path - str - Path to directory containing Galaxia3 and isoPARSEC folders - e.g. '/home/user/PATHTOFOLDER/'
        photo_path - str - Path to folder containing photometric survey data - e.g. '/PATH/Galaxia3/photometric'

        spectro - str - Name of spectroscopic survey - e.g. 'Galaxia3'
        spectro_fname - str - Name of file containing spectroscopic survey data - e.g. 'Galaxia3_survey.csv'
        field_fname - str - data file for field pointings - e.g. 'Galaxia3_fieldinfo.csv'

        spectro_coords - list of str - List of column headers taken from spectroscopic survey - 
                    - e.g. ['fieldid', 'glon', 'glat', 'Happ', 'Japp', 'Kapp']
        field_coords - tuple of list and str - List of column headers taken from pointings file and coord system (Equatorial or Galactic)
                    - e.g. ['fieldID', 'glon', 'glat', 'Magmin', 'Magmax', 'Colmin', 'Colmax']
        photo_coords - list of str - List of column headers taken from photometric survey
                    - e.g. ['glon', 'glat', 'Happ', 'Japp', 'Kapp']

        spectro_dtype - list of dtypes - data types of columns in spectroscopic catalogue
                    - e.g. [str, float, float, float, float, float]
        field_dtype - list of dtypes - data types of columns in field info file
                    - e.g. [str, float, float, float, float, float, float]
        photo_dtype - list of dtypes- data types of columns in photometric catalogue
                    - e.g. [float, float, float, float, float]

        photo_pickle_fname - str - File name for pickled instance of photometric interpolants - e.g. '/2massdata_RAVEfields.pickle'
        spectro_pickle_fname - str - File name for pickled instance of spectroscopic interpolants - e.g. '/2massdata_RAVEfields.pickle'
        sf_pickle_fname - str - File name for pickled instance of physical selection function interpolants - e.g. '/2massdata_RAVEfields.pickle'

        photo_tag - str - File type tagged on the end of photometric data folders - e.g. '.csv'
        fieldlabel_type - obj - Data type used for names of fields in the pointings catalogue - e.g. str

        iso_pickle_file - str - File name for pickled isochrones instances - e.g. "/evoTracks/isochrone_distributions_resampled.pickle" 

    Example:
    --------

        fileinfo_path = '/home/user/..../Galaxia3/Galaxia3_fileinfo.pickle'

        # Load in class instance
        Demo = surveyInfoPickler.surveyInformation()
        Demo.load(fileinfo_path)

        # See attributes of class
        Demo.printValues()

        # Change incorrect attributes
        Demo.attribute = ReplacementAttribute # e.g. new path to data or new file location

        # Run the __call__ routine to setup the file locations
        Demo()

        # testFiles checks whether the information given is accurate
        # If this is the first time running, pickle files shouldn't exist
        Demo.testFiles()

        # Pickle the file information
        Demo.save(fileinfo_path)
        '''

    def __init__(self, path=None, locQ=True):

        self.style = ''

        self.data_path = ''
        self.survey = ''
        self.survey_folder = os.path.join(self.data_path, self.survey)

        self.photo_path = ''

        self.spectro_fname = ''
        self.spectro_path = os.path.join(self.survey_folder, self.spectro_fname)

        self.field_fname = ''
        self.field_path = os.path.join(self.survey_folder, self.field_fname)

        self.spectro_coords = None
        self.field_coords = None
        self.photo_coords = None
        self.spectro_dtypes = None
        self.field_dtypes = None
        self.photo_dtypes = None
        self.coord_system = ''
        self.theta_rng = None
        self.phi_rng = None

        self.sf_pickle_fname = ''
        self.sf_pickle_path = os.path.join(self.survey_folder, self.sf_pickle_fname)
        self.obsSF_pickle_fname = ''
        self.obsSF_pickle_path = os.path.join(self.survey_folder, self.obsSF_pickle_fname)

        self.photo_tag = None
        self.fieldlabel_type = None #self.spectro_dtypes[0]

        self.iso_folder = ''
        self.iso_pickle_file = ''
        self.iso_pickle_path = os.path.join(self.survey_folder, self.iso_folder, self.iso_pickle_file)
        self.iso_interp_file = ''

        self.fileinfo_path = ''

        self.photo_field_files = ['']
        self.photo_field_paths = ['']
        self.photo_field_starcount = {}

        self.photo_model = ()
        self.spectro_model = ()

        if path is not None:
            path = os.path.abspath(path)
            self.load(path, locQ=locQ)

    def __call__(self):

        self.survey_folder = os.path.join(self.data_path, self.survey)
        self.spectro_path = os.path.join(self.survey_folder, self.spectro_fname)

        self.field_path = os.path.join(self.survey_folder, self.field_fname)

        self.photo_path = os.path.join(self.survey_folder, 'photometric')

        self.sf_pickle_path = os.path.join(self.survey_folder, self.sf_pickle_fname)
        self.obsSF_pickle_path = os.path.join(self.survey_folder, self.obsSF_pickle_fname)

        self.iso_data_path = os.path.join(self.survey_folder, self.iso_folder, self.iso_data_file)
        self.iso_interp_path = os.path.join(self.survey_folder, self.iso_folder, self.iso_interp_file)

        self.photo_field_paths = {field: os.path.join(self.photo_path, fieldfile) for field, fieldfile in self.photo_field_files.iteritems()}

        self.example_string = \
"""
# Get file names and coordinates from pickled file
pickleFile = '{directory}/{label}/{label}_fileinfo.pickle'\n

# Load in class instance
{label} = surveyInfoPickler.surveyInformation()
{label}.load(pickleFile)\n

# See attributes of class
{label}.printValues()

# Change incorrect attributes
{label}.attribute = ReplacementAttribute # e.g. new path to data or new file location

# Run the __call__ routine to setup the file locations
{label}()

# testFiles checks whether the information given is accurate
# If this is the first time running, pickle files shouldn't exist
{label}.testFiles()

# Pickle the file information
{label}.save(fileinfo_path)
""".format(label=self.survey, directory=self.data_path)

        # Set the class instance doc string as a coding example
        self.__doc__ = self.example_string

    def pickleInformation(self, filename):

        # Can you pickle a class from inside the class?
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

    def save(self):

        self.saveas(self.fileinfo_path)

    def saveas(self, filename):

        # Convert attributes to dictionary
        attr_dict = vars(self)

        # Dump pickled dictionary of attributes
        with open(filename, 'wb') as handle:
            pickle.dump(attr_dict, handle, protocol=2)

    def load(self, filename, locQ=True):

        # Load pickled dictionary of attributes
        with open(filename, "rb") as f:
            file_dict  = pickle.load(f) 
        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])

        if locQ:

            # If the path to the data has changed
            if filename != self.fileinfo_path:

                # False until we get a y/n response
                reset = None
                
                # Now enter while loop
                while not reset in ('y','n'):
                    print(reset)
                    reset = raw_input("File location has changed, reset the file locations? (y/n)")
                    print(reset)
                # Reset file names to correct path
                if reset == 'y': 
                    relocate(filename)

                    # Load new pickled dictionary of attributes
                    with open(filename, "rb") as f:
                        file_dict  = pickle.load(f) 
                    # Convert dictionary to attributes  
                    for key in file_dict:
                        setattr(self, key, file_dict[key])

    def testFiles(self):

        # Try to open folders and check file names and data structures

        # 1) Test paths - photo_path, spectro_path, field_path
        print("1) Checking file paths exist:")
        good = True
        if not os.path.exists(self.spectro_path):
            print("The path to sprectrograph catalogue (spectro_path) does not exist: %s" % self.spectro_path)
            good = False
        if not os.path.exists(self.field_path):
            print("The path to field info file (field_path) does not exist: %s" % self.field_path)
            good = False
        if not os.path.exists(self.photo_path):
            print("The path to photometric folder (photo_path) does not exist: %s" % self.photo_path)
            good = False
        if good: print("OK")
        print('')

        # 2) Test spectro file - column headers, data types
        print("2) Checking spectroscopic catalogue file structure:")
        good = True
        if os.path.exists(self.spectro_path):
            try:
                # Load in dataframe
                df = pd.read_csv(self.spectro_path)
                df_in = True
            except ValueError:
                # Raise error if dataframe cannot be loaded in.
                print("pd.read_csv('%s') failed for some reason. Please try to fix this." % self.spectro_path)
                df_in = False
                good = False
            if df_in:
                if not set(self.spectro_coords).issubset(list(df)):
                    # If column headers don't match spectro_coords, return error
                    print("Column headers are %s, \nbut spectro_coords suggests %s, \nplease resolve this.\n" %  (df.columns.values, self.spectro_coords))
                    good = False
                    if self.style == 'as': print("(Before running HealpixAssignment there won't be a 'fieldID' column)")
                else:
                    for i in range(len(self.spectro_coords)):
                        # Check that each coordinate has the right datatype.
                        if not df[self.spectro_coords[i]].dtype == self.spectro_dtypes[i]:
                            print("Datatype of column %s given as %s but is actually %s." % (self.spectro_coords[i], str(self.spectro_dtypes[i]), str(df[self.spectro_coords[i]].dtype.type)))
                            self.spectro_dtypes[i] = df[self.spectro_coords[i]].dtype.type
                            print("Changed dtype to %s, run self.save() to keep these changes." % df[self.spectro_coords[i]].dtype.type)
                            good = False
                    # Check that longitude and latitude are in the right range
                    theta = df[self.spectro_coords[2]]
                    phi = df[self.spectro_coords[1]]
                    inrange = all(theta>=self.theta_rng[0])&all(theta<=self.theta_rng[1])&all(phi>=self.phi_rng[0])&all(phi<=self.phi_rng[1])
                    if not inrange:
                        print("spectro_path angle range not correct. Should be -pi/2<=theta<=pi/2, 0<=phi<=2pi. Data gives %s<=theta=<%s, %s<=phi<=%s." % \
                            (str(min(theta)), str(max(theta)), str(min(phi)), str(max(phi))))
                        good = False
        if good: print("OK")
        print('')

        # 3) Test field file - column headers, data types
        print("3) Checking field information file structure:")
        good = True
        if os.path.exists(self.field_path):
            try:
                # Load in dataframe
                df = pd.read_csv(self.field_path)
                df_in = True
            except ValueError:
                # Raise error if dataframe cannot be loaded in.
                print("pd.read_csv('%s') failed for some reason. Please try to fix this." % self.field_path)
                good = False
                df_in = False
            if df_in:
                if not set(self.field_coords).issubset(list(df)):
                    # If column headers don't match field_coords, return error
                    print("Column headers are %s, \nbut field_coords suggests %s, \nplease resolve this.\n" %  (df.columns.values, self.field_coords))
                    good = False
                else:
                    for i in range(len(self.field_coords)):
                        # Check that each coordinate has the right datatype.
                        if not df[self.field_coords[i]].dtype == self.field_dtypes[i]:
                            print("Datatype of column %s given as %s but is actually %s." % (self.field_coords[i], str(self.field_dtypes[i]), str(df[self.field_coords[i]].dtype.type)))
                            self.field_dtypes[i] = df[self.field_coords[i]].dtype.type
                            print("Changed dtype to %s, run self.save() to keep these changes." % df[self.field_coords[i]].dtype.type)
                            good = False

                    if self.style == 'mf':
                        # Check that longitude and latitude are in the right range
                        theta = df[self.field_coords[2]]
                        phi = df[self.field_coords[1]]
                        inrange = all(theta>=self.theta_rng[0])&all(theta<=self.theta_rng[1])&all(phi>=self.phi_rng[0])&all(phi<=self.phi_rng[1])
                        if not inrange:
                            print("field_path angle range not correct. Should be -pi/2<=theta<=pi/2, 0<=pi<=2pi. Data gives %s<=theta=<%s, %s<=phi<=%s." % \
                                (str(min(theta)), str(max(theta)), str(min(phi)), str(max(phi))))
                            good = False
                        # Check that half-angle is in range
                        halfangle = df[self.field_coords[3]]
                        inrange = all(halfangle>=0)&all(halfangle<=np.pi)
                        if not inrange:
                            print("Halfangle out of range. Should be 0<=halfangle<=pi. Data gives %s<=halfangle=<%s." % \
                                (str(min(halfangle)), str(max(halfangle))))
                            good = False
                        print("(make sure halfangle is in units of radians.)")
        if good: print("OK")        
        print('')

        # 4) Test photo file - column headers, data types
        print("4) Checking photometric catalogue file structure:")
        good = True
        if os.path.exists(self.photo_path):
            # Find all photometric files in folder
            photo_files = os.listdir(self.photo_path)
            # Pick a random file:
            photo_file = photo_files[np.random.randint(len(photo_files))]
            print("Checking %s:" % photo_file)
            # Join with photo_path to find path to files
            filepath = os.path.join(self.photo_path, photo_file)
            try:
                # Load in dataframe
                df = pd.read_csv(filepath, nrows=3)
                df_in = True
            except ValueError:
                # Raise error if dataframe cannot be loaded in.
                print("pd.read_csv('%s') failed for some reason. Please try to fix this." % filepath)
                df_in = False
                good = False
            if df_in:
                if not set(self.photo_coords).issubset(list(df)):
                    # If column headers don't match photo_coords, return error
                    print("Column headers are %s, \nbut photo_coords suggests %s, \nplease resolve this.\n" %  (df.columns.values, self.photo_coords))
                    good = False
                else:
                    for i in range(len(self.photo_coords)):
                        # Check that each coordinate has the right datatype.
                        if not df[self.photo_coords[i]].dtype == self.photo_dtypes[i]:
                            print("Datatype of column %s given as %s but is actually %s." % (self.photo_coords[i], str(self.photo_dtypes[i]), str(df[self.photo_coords[i]].dtype.type)))
                            self.photo_dtypes[i] = df[self.photo_coords[i]].dtype.type
                            print("Changed dtype to %s, run self.save() to keep these changes." % df[self.photo_coords[i]].dtype.type)
                            good = False
                    # Check that longitude and latitude are in the right range
                    theta = df[self.photo_coords[1]]
                    phi = df[self.photo_coords[0]]
                    inrange = all(theta>=self.theta_rng[0])&all(theta<=self.theta_rng[1])&all(phi>=self.phi_rng[0])&all(phi<=self.phi_rng[1])
                    if not inrange:
                        print("photo_path angle range not correct. Should be -pi/2<=theta<=pi/2, 0<=pi<=2pi. Data gives %s<=theta=<%s, %s<=phi<=%s." % \
                            (str(min(theta)), str(max(theta)), str(min(phi)), str(max(phi))))
                        good = False
        else: 
            print("Path to folder of photometric files, %s, not found." % self.photo_path)
            good = False
        if good: print("OK")            
        print('')

        # 5) Test SF paths
        print("5) Checking selection function pickle paths exist:")
        good = True
        if not os.path.exists(self.sf_pickle_path):
            print("The path to your selection function pickled instance, sf_pickle_path, does not exist: %s" % self.sf_pickle_path)
            good = False
        if not os.path.exists(self.obsSF_pickle_path):
            print("The path to your selection function pickled instance, obsSF_pickle_path, does not exist: %s" % self.obsSF_pickle_path)
            good = False
        if not good: 
            print("^ These files should exist for an already made selection function. If you're starting from scratch, ignore this!")
        else: print("OK")
        print('')

        # 6) Test isochrone paths
        print("6) Checking isochrone pickle files exist:")
        good = True
        if not os.path.exists(self.iso_interp_path):
            print("The path to isochrone data, iso_interp_path, does not exist: %s" % self.iso_interp_path)
            if not os.path.exists(self.iso_data_path):
                print("The path to isochrone data, iso_data_path, does not exist: %s" % self.iso_data_path)
                print("(At lease one of the above files mus exist to generate a selection function in intrinsic coordinates)")
                good = False
            else: print("The selection function will generate new isochrone interpolants using data from %s." % self.iso_data_file)
        else: print("The premade interpolants (%s) will be automatically be used to calculate the selection function." % self.iso_interp_file)

    def photoTest(self, photo_files):

        '''
        photoTest - Test photometric files for field assignment to check data ranges and column headers etc.

        Parameters
        ----------
            photo_files: list of str
        
        **kwargs
        --------


        Returns
        -------


        '''

        # 4) Test photo file - column headers, data types
        print("Checking photometric catalogue file structure:")
        good = True

        # Pick a random file:
        photo_file = photo_files[np.random.randint(len(photo_files))]
        print("Checking %s:" % photo_file)
        if os.path.exists(photo_file):
            # Join with photo_path to find path to files
            filepath = os.path.join(photo_file)
            try:
                # Load in dataframe
                df = pd.read_csv(filepath, nrows=3)
                df_in = True
            except ValueError:
                # Raise error if dataframe cannot be loaded in.
                print("pd.read_csv('%s') failed for some reason. Please try to fix this." % filepath)
                df_in = False
                good = False
            if df_in:
                if not set(self.photo_coords).issubset(list(df)):
                    # If column headers don't match photo_coords, return error
                    print("Column headers are %s, \nbut photo_coords suggests %s, \nplease resolve this.\n" %  (df.columns.values, self.photo_coords))
                    good = False
                else:
                    for i in range(len(self.photo_coords)):
                        # Check that each coordinate has the right datatype.
                        if not df[self.photo_coords[i]].dtype == self.photo_dtypes[i]:
                            print("Datatype of column %s given as %s but is actually %s." % (self.photo_coords[i], str(self.photo_dtypes[i]), str(df[self.photo_coords[i]].dtype.type)))
                            self.photo_dtypes[i] = df[self.photo_coords[i]].dtype.type
                            print("Changed dtype to %s, run self.save() to keep these changes." % df[self.photo_coords[i]].dtype.type)
                            good = False
                    # Check that longitude and latitude are in the right range
                    theta = df[self.photo_coords[1]]
                    phi = df[self.photo_coords[0]]
                    inrange = all(theta>=self.theta_rng[0])&all(theta<=self.theta_rng[1])&all(phi>=self.phi_rng[0])&all(phi<=self.phi_rng[1])
                    if not inrange:
                        print("photo_path angle range not correct. Should be -pi/2<=theta<=pi/2, 0<=pi<=2pi. Data gives %s<=theta=<%s, %s<=phi<=%s." % \
                            (str(min(theta)), str(max(theta)), str(min(phi)), str(max(phi))))
                        good = False
            if good: 
                print("File OK")    
                forward_bool = True
            else: 
                good_response = False
                while not good_response:
                    forward = raw_input("Tests on the files have raised some warnings. Would you like to continue anyway? (y/n)")        
                    if forward == 'n': 
                        forward_bool = False
                        good_response = True
                    elif forward == 'y':
                        forward_bool = True
                        good_response = True
                    else: pass # Bad response to input

        else: # Photometric file not found
            forward_bool=False
            print("Path to folder of photometric files, %s, not found." % photo_file)
        print('')

        return forward_bool

    def printValues(self):

        print("Location of spectroscopic catalogue")
        print("spectro_path: " + self.spectro_path)
        print("Filename (in Demo.spectro file) for field pointings")
        print("field_path: " + self.field_path +'\n')
        print("Location of photometric folder of field files")
        print("photo_path: " + self.photo_path)

        print("""Column headers and dtypes for spectrograph information [ fieldID, Phi, Th, magA, magB, magC]
where magA-magB = Colour, magC = m (for selection limits)""")
        print("spectro_coords: " + str(self.spectro_coords))
        print("spectro_dtypes: " + str(self.spectro_dtypes))
        print("Column headers for field pointing information [fieldID, Phi, Th, halfangle, Magmin, Magmax, Colmin, Colmax]")
        print("field_coords: " + str(self.field_coords))
        print("field_dtypes: " + str(self.field_dtypes))
        print("""Column headers and dtypes for photometric field files [ Phi, Th, magA, magB, magC]
where magA-magB = Colour, magC = m (for selection limits)""")
        print("photo_coords: " + str(self.photo_coords))
        print("photo_dtypes: " + str(self.photo_dtypes))

        print('Coordinate system of angles ("Equatorial" or "Galactic")')
        print("coord_system:" + str(self.coord_system) +'\n')

        print("pickled file locations which will store the selection function information:")
        print("sf_pickle_path: " + self.sf_pickle_path)
        print("obsSF_pickle_path: " + self.obsSF_pickle_path +'\n')

        print("File types for photometric data")
        print("photo_tag: " + str(self.photo_tag))

        print("File containing dill instance of isochrone data.")
        print("iso_data_file: " + self.iso_data_path)
        print("File containing pickled isochrone interpolants.")
        print("iso_interp_path: " + self.iso_interp_path)
        print("File containing pickled isochrone magnitudes.")
        print("iso_mag_path: " + self.iso_mag_path +'\n')

    def example(self):

        print(self.example_string)

def relocate(fileinfo_path):

    fileinfo = surveyInformation(fileinfo_path, locQ=False)

    # Change name of fileinfo_path
    fileinfo.fileinfo_path = fileinfo_path

    # Tuple of steps to file location
    root = fileinfo_path.split("/")[:-2]
    # Directory in which data is stored
    directory = "/".join(root)

    # Replace directory with the correct one
    fileinfo.data_path = directory
    # Update remaining entries
    fileinfo()
    
    # Repickle file
    fileinfo.save()