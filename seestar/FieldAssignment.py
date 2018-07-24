'''
FieldAssignment - Contains classes for assigning stars from all-sky surveys to individual fields

Classes
-------
FieldAssignment - For assigning stars from an allsky survey to the field pointings of a multi-fibre survey.

HealpixAssignment - For assigning stars from a allsky surveys HEALPix pixels on an array.

Functions
---------
importLimit - Number of rows which can be imported at once whilst using a set amount of memory.

iterLimit - Number of stars which may be assigned at once without using too much memory
    (Field assignment is quite memory intensive as it uses Nfield*Nstar matrices of booleans and strings)

countStars - iterate through files to find the total number of photometric stars

labelStars - Gives each star a label according to the pixel which it will be assigned to

numberPixels - Find a good number of pixels to use for survey

Requirements
------------

ArrayMechanics.py
surveyInfoPickler.py
'''

import numpy as np
import pandas as pd
import healpy as hp

import gc
import cProfile, pstats, time

import os, sys
import gzip
import psutil
from shutil import copyfile

from seestar import ArrayMechanics
from seestar import surveyInfoPickler

class FieldAssignment():

    '''
    FieldAssignment - For assigning stars from an allsky survey to the field pointings of a multi-fibre survey.
        - Generate a file for each field in the survey.

    args
    ----
        fileinfo_path: string
            - surveyInformation pickle instance which includes the required format for files

      photometric_files: list of str
          - Paths to all files with the photometric data
          - If only one datafile, set only one element of list equal to that filename

    Functions
    ---------
        ClearFiles() - Clears all field pointing files and leaves headers
        RunAssignmentAPTPM(): - Runs thorough method for calculating field assignments

    Demo
    ----
        # Load in fileinfo class instance
        fileinfo = surveyInfoPickler.surveyInformation('/PATH/TO/DATA/test/test_fileinfo.pickle')
        # List of files containing photometric catalogue data
        allsky_files = ["/media/andy/UUI/Research/SF/SFdata/Galaxia/Galaxia3/Galaxy3_file"+str(n)+".csv" for n in range(1, 15)]
        # Run field assignment to produce photometric field files
        assign = FieldAssignment.FieldAssignment(fileinfo, photometric_files=allsky_files)
        assign()
    '''

    def __init__(self, fileinfo_path,
                 photometric_files):

        # surveyInformation instance
        self.fileinfo = surveyInfoPickler.surveyInformation(fileinfo_path)

        # Test photometric file information - continue if forward is true
        forward = self.fileinfo.photoTest(photometric_files)
        if not forward: print("Resolve issues in the data then run again.")
        else:
            # List of files with full galaxy data
            # All files should be .csv format
            self.photometric_files = photometric_files

            # Total number of stars in database
            self.total = 470000000#countStars(photometric_files)
            
            # Setup pointings dataframe containing unique field positions
            pointings_path = self.fileinfo.field_path
            pointings = pd.read_csv(pointings_path)
            # Make pointings a class attribute
            self.pointings = pointings

            # ID type for fields in pointings
            self.IDtype = self.fileinfo.field_dtypes[0]

            # Number of stars to be imported at once
            self.N_import = importLimit(photometric_files, proportion=0.05)

            # Number of stars to be iterated over
            Npoint = len(pointings)
            self.N_iter = iterLimit(Npoint, proportion=0.05)

            # Output information on processing
            print("Total number of stars %d." % self.total)
            print("Importing {Nimport} stars at a time. Iterating {Niter} stars at a time.".format(Nimport=self.N_import, Niter=self.N_iter))

            files = [os.path.join(self.fileinfo.photo_path, str(field))+'.csv' for field in self.pointings[self.fileinfo.field_coords[0]]]
            print("Field file path for field {}: {}".format(self.pointings[self.fileinfo.field_coords[0]][0], files[0]))

            self.__call__()

    def __call__(self):

        self.ClearFiles()
        self.RunAssignmentAPTPM(N_import=self.N_import, N_iterate=self.N_iter)

    def ClearFiles(self):

        '''
        ClearFiles() - Clears all field pointing files and leaves headers

        Inherits
        --------
            photometric_files - Full photometric data

            fileinfo - class instance of surveyInformation
        '''
        
        print('Clearing field files...')
        
        # Take headers from source files (photometric catalogue)
        headers = pd.read_csv(self.photometric_files[0], nrows = 1)[:0]
        
        # Write all field files with the appropriate headers
        for field in self.pointings[self.fileinfo.field_coords[0]]:
            headers.to_csv(os.path.join(self.fileinfo.photo_path, str(field))+'.csv',
                          index=False, mode = 'w')
            
        print('...done\n')
        

    def RunAssignmentAPTPM(self, N_import=100000, N_iterate=1000):

        '''
        RunAssignmentAPTPM - Assigns stars in photometric catalogue to field files

        **kwargs
        --------
            N_import=100000: int
                - Number of stars which can be imported at once without using a lot of memory

            N_iterate=1000: int
                - Numebr of stars which can be iterated in the method at once without using a lot of memory

        '''
            
        # Time in order to track progress
        start = time.time()
        # For analysing the progress
        starsanalysed = 0

        # Photometric file names
        field_files = {field: str(field)+'.csv' for field in self.pointings[self.fileinfo.field_coords[0]]}

        # Open files for writing
        open_files = {}
        for field in self.pointings[self.fileinfo.field_coords[0]]:
            open_files[field] = open(os.path.join(self.fileinfo.photo_path, field_files[field]), 'a+')

        # Count dictionary for number of stars per field (all entries start at 0)
        starcount = {field: 0 for field in self.pointings[self.fileinfo.field_coords[0]]}
        total = 0
        outString = ""

        # Iterate over full directory files
        for filename in self.photometric_files:
            
            for df_allsky in pd.read_csv(filename, chunksize=N_import, low_memory=False):
            
                starsanalysed += len(df_allsky)
                fname = filename.split('/')[-1]
                    
                # Column header labels in pointings
                phi, theta, halfangle = self.fileinfo.field_coords[1:4]
                    
                df_allsky = ArrayMechanics.\
                            AnglePointsToPointingsMatrix(df_allsky, self.pointings, phi, theta, halfangle,
                                                        IDtype = self.IDtype, Nsample = N_iterate, 
                                                        progress=True, outString=outString)
                
                field_i=0
                for field in self.pointings[self.fileinfo.field_coords[0]]:
                    field_i += 1
                    # Write the number of fields which have been saved so far
                    sys.stdout.write('\r'+outString+'...Saving: '+str(field_i)+'/'+str(len(self.pointings)))

                    # Check which rows are assigned to the right field
                    df_bool = df_allsky.points.apply(lambda x: field in x)
                    df = df_allsky[df_bool].copy()

                    # Add to star count
                    starcount[field] += len(df)
                    total += len(df)

                    df.drop('points', inplace=True, axis=1)
                    
                    df.to_csv(open_files[field], index=False, header=False)

                # Updates of progress continuously output
                perc = round((starsanalysed/float(self.total))*100, 3)
                duration = round((time.time() - start)/60., 1)
                projected = round((time.time() - start)*self.total/((starsanalysed+1)*3600), 3)
                hours = int(projected)
                minutes = int((projected - hours)*60)
                outString = '\r'+'File: '+fname+'  '+\
                               'Complete: '+str(starsanalysed)+'/'+str(self.total)+'('+\
                               str(perc)+'%)  Time: '+str(duration)+'m  Projected: '+str(hours)+'h'+str(minutes)+'m'

        print("\nTotal stars assigned to fields: %d.\n\
Dictionary of stars per field in fileinfo.photo_field_starcount." % total)

        self.fileinfo.photo_field_files = field_files
        self.fileinfo.photo_field_starcount = starcount
        self.fileinfo()
        self.fileinfo.save()

def importLimit(files, proportion=0.1):

    '''
    importLimit - Number of rows which can be imported at once whilst using a set amount of memory.

    Parameters
    ----------
        files: list of str
            - file paths for all files containing photometric catalogue

    **kwargs
    --------
        proportion: float (<1)
            - Proportion of available memory which this can use

    Returns
    -------
        import_max: int
            - Number of rows to be imported at once
    '''

    filename = files[0]
    df = pd.read_csv(filename, nrows=1000)

    mem_thousand = sys.getsizeof(df)
    mem = psutil.virtual_memory()

    ratio = mem.available/mem_thousand
    import_max = 1000 * ratio * proportion
    
    return int(import_max)

def iterLimit(Npoint, proportion=0.1):

    '''
    iterLimit - Number of stars which may be assigned at once without using too much memory
        (Field assignment is quite memory intensive as it uses Nfield*Nstar matrices of booleans and strings)

    Parameters
    ----------
        Npoint: int
            - Number of fields in the survey to be assigned

    **kwargs
    --------
        proportion: float (<1)
            - Proportion of available memory which this can use

    Returns
    -------
        iter_max: int
            - Number of rows to be iterated at once
    '''

    mem = psutil.virtual_memory()

    iter_max = (mem.available * proportion)/(50 * Npoint)
    
    return int(iter_max)

def countStars(files):

    '''
    countStars - iterate through files to find the total number of photometric stars

    Parameters
    ----------
        files: list of str
            - All photometric files containing catalogue data

    Returns
    -------
        count: int
            - Total number of stars in photometric catalogue
    '''

    print "Counting total number of stars",

    count = 0
    for filen in files:
        print ".",
        extension = os.path.splitext(filen)[-1]
        if extension=='.gz':
            with gzip.open(filen) as f:
                for _ in f:
                    count += 1         
        else:
            with open(filen) as f:
                for _ in f:
                    count += 1         

    print("done")
    return count

class HealpixAssignment():

    '''
    HealpixAssignment - Assign all-sky survey stars to pixels in HEALPix array.
        - Generate a fieldinfo file for the pixels
        - Add field id to each star in the spectroscopic catalogue
        - Assign stars in photometric catalogue to pixel field files

    args
    ----
        fileinfo_path: string
            - surveyInformation pickle instance which includes the required format for files

        photometric_files: list of str
          - Paths to all files with the photometric data
          - If only one datafile, set only one element of list equal to that filename

        npixel: int
          - Number of pixels wanted in the HEALPix array.

    Functions
    ---------
        fieldinfo_file - Generate field info file with all fields (pixels) listed with colour and magnitude limits.

        spectroPixels - Adds fieldID column to spectroscopic catalogue
        
        photoPixelFiles - Assign photometric catalogue stars to pixel field files

    Demo
    ----
        # Load in fileinfo class instance
        fileinfo = surveyInfoPickler.surveyInformation('/PATH/TO/DATA/test/test_fileinfo.pickle')
        # List of files containing photometric catalogue data
        allsky_files = ["/PATH/TO/DATA/photometric_file"+str(n)+".csv" for n in range(1, 15)]
        # Number of HEALPix pixels
        npixel = 1000
        # Run field assignment to produce photometric field files
        assign = healpixFields.HealpixAssignment(fileinfo_path, photometric_files, npixel)
        assign()
    '''

    def __init__(self, fileinfo_path,
                photometric_files, npixel):

        # surveyInformation instance
        self.fileinfo = surveyInfoPickler.surveyInformation(fileinfo_path)

        # List of files with full galaxy data
        # All files should be .csv format
        self.photometric_files = photometric_files

        self.phi = self.fileinfo.spectro_coords[1]
        self.theta = self.fileinfo.spectro_coords[2]
        self.rng_th = self.fileinfo.theta_rng
        self.rng_phi = self.fileinfo.phi_rng

        # number of pixels to be used
        self.npixel = numberPixels(npixel)
        self.nside = hp.npix2nside(self.npixel)

        # Total number of stars in database
        self.total = countStars(photometric_files)

        # Directory and names of pointings files
        self.fieldpoint_directory = self.fileinfo.photo_path

        # Number of stars to be imported at once
        self.N_import = importLimit(photometric_files)

    def __call__(self):

        self.fieldinfo_file()
        self.spectroPixels()
        self.photoPixelFiles()

    def fieldinfo_file(self):

        '''
        fieldinfo_file - Generate field info file with all fields (pixels) listed with colour and magnitude limits.

        Inherited
        ---------
            npixel: int
                - Number of pixels in the HEALPix array

            fileinfo: surveyInformation instance
                - file information for catalogue
        '''

        # List of field numbers
        fields = np.arange( self.npixel )
        # Generate the dataframe for field info
        df = pd.DataFrame(fields.T, columns=[self.fileinfo.field_coords[0]])

        # Take magnitude and colour limits as input
        good_mag = False
        while not good_mag:
            try:
                # Take each value as an input from the user
                mag_l = input("Survey lower limit on H-band apparent magnitude (if not given, type None): ")
                mag_u = input("Survey upper limit on H-band apparent magnitude (if not given, type None): ")
                col_l = input("Survey lower limit on J-K colour (if not given, type None): ")
                col_u = input("Survey upper limit on J-K colour (if not given, type None): ")
            except NameError:
                # If a non-assigned object input is used
                print("Return float or None.")
                
            if ((mag_l<mag_u) | (mag_l==None) | (mag_u==None)) & \
                ((col_l<col_u) | (col_l==None) | (col_u==None)):
                # If upper>lower limits or either upper or lower are None then we can continue.
                # TBH I think we can do better than this so it's a work in progress
                good_mag = True
            elif not (mag_l<mag_u):
                print("magnitude limits don't make sense.")
            elif not (col_l<col_u):
                print("colour limits don't make sense.")

        # Generate a magnitude lower limit column
        if mag_l == None: df[self.fileinfo.field_coords[1]] = 'NoLimit'
        else: df[self.fileinfo.field_coords[1]] = mag_l
        # Generate a magnitude upper limit column
        if mag_u == None: df[self.fileinfo.field_coords[2]] = 'NoLimit'
        else: df[self.fileinfo.field_coords[2]] = mag_u
        # Generate a colour lower limit column
        if col_l == None: df[self.fileinfo.field_coords[3]] = 'NoLimit'
        else: df[self.fileinfo.field_coords[3]] = col_l
        # Generate a colour upper limit column
        if col_u == None: df[self.fileinfo.field_coords[4]] = 'NoLimit'
        else: df[self.fileinfo.field_coords[4]] = col_u

        # Save the dataframe to the field info file
        df.to_csv(self.fileinfo.field_path, index=False)

    def spectroPixels(self):

        '''
        spectroPixels - Adds fieldID column to spectroscopic catalogue

        Inherited
        ---------
            nside: int
                - Parameter of HEALPix which determines the structure of the array
                - npixel = 12*(nside**2)

            fileinfo: surveyInformation instance
                - file information for catalogue

            self.theta, self.phi: str
                - Names of theta and phi columns in dataframe (from fileinfo.spectro_coords)

            rng_th, rng_phi: tupples of floats
                - Ranges of theta and phi coordinates, e.g. (-pi/2, pi/2), (0, 2pi)
        '''

        # Load in spectroscopic catalogue
        df = pd.read_csv(self.fileinfo.spectro_path)

        # Set fieldID column using labelStars
        df['fieldID'] = labelStars(df[self.theta], df[self.phi], self.rng_th, self.rng_phi, self.nside)

        # Assign 'fieldID' name to fileinfo.spectro_coords
        self.fileinfo.spectro_coords[0] = 'fieldID'
        self.fileinfo.save()

        # Save csv file
        df.to_csv(self.fileinfo.spectro_path, index=False)      

    def photoPixelFiles(self):
        # Put photometric  files into pixel files

        '''
        photoPixelFiles - Assign photometric catalogue stars to pixel field files

        Inherited
        ---------
            photometric_files: list of str
                - Paths to files containing photometric catalogue data

            N_import: int
                - Number of stars which can be imported at one time

            total: int
                - The total number of photometric catalogue

            npixel: int
                - Number of pixels in the HEALPix array

            nside: int
                - Parameter of HEALPix which determines the structure of the array
                - npixel = 12*(nside**2)

            fileinfo: surveyInformation instance
                - file information for catalogue

            self.theta, self.phi: str
                - Names of theta and phi columns in dataframe (from fileinfo.spectro_coords)

            rng_th, rng_phi: tupples of floats
                - Ranges of theta and phi coordinates, e.g. (-pi/2, pi/2), (0, 2pi)
        '''

        # Number of stars analysed so far in order to keep an eye on progress
        starsanalysed = 0
        # Boolean for if this is the first file so if headers need to be saved
        firstfile = True

        # Iterate through photometric files
        for filename in self.photometric_files:
            # Run through data in N_import sized chunks
            for df in pd.read_csv(filename, chunksize=self.N_import):
                # Increase stars analysed for each chunk which is imported
                starsanalysed += len(df_allsky)
                # Find pixel assignments of stars in catalogue
                pix_num = labelStars(df[self.theta], df[self.phi], self.rng_th, self.rng_phi, self.nside)
                # Iterate through pixels to assign to files
                for pix in np.arange(self.npixel):
                    # Name of the file being imported
                    fname = filename.split('/')[-1]
                    # Updates of progress continuously output
                    perc = round((starsanalysed/float(self.total))*100, 3)
                    sys.stdout.write('\r'+'allsky file: '+fname+'  '+\
                                   'Completion: '+str(starsanalysed)+'/'+str(self.total)+'('+
                                   str(perc)+'%)  fieldID: '+str(pix))
                    sys.stdout.flush()

                    # Find stars which are on the given pixel
                    field_stars = df[pix_num == pix]
                    # Filename for df to be saved to
                    fname = os.path.join(self.fileinfo.photo_path, str(pix)) + '.csv'
                    if firstfile: # Add headers and write over file if this is the first file
                        field_stars.to_csv(fname, index=False)
                    else: # No headers and append to data currently there
                        field_stars.to_csv(fname, mode='a', header=False, index=False)
                # Once all files have headers set firstfile is turned to false
                firstfile=False

def labelStars(theta, phi, rng_th, rng_phi, nside):

    '''
    labelStars - Gives each star a label according to the pixel which it will be assigned to

    Parameters
    ----------
        theta: arr of floats
            - theta (glat, Dec) coordinates of stars being assigned

        phi: arr of floats
            - phi (glon, RA) coordinates of stars being assigned

        rng_th: tuple of floats
            - Range of values of theta (e.g. (-pi/2, pi/2))

        rng_phi: tuple of floats
            - Range of values of phi (e.g. (0, 2pi))
    
        nside: int
            - Parameter of HEALPix (npixel = 12*nside**2)

    The function corrects the theta, phi range to (-pi/2, pi/2), (-pi, pi).

    Returns
    -------
        pixel: arr of int
            - Pixel number for each star in the array of stars

    '''

    # Need to correct the range: 0<th<pi, -pi<phi<pi
    theta1 = 0 + ( (theta - rng_th[0]) * (np.pi)/(rng_th[1]-rng_th[0]) )
    phi1 = -np.pi + ( (phi - rng_phi[0]) * (2*np.pi)/(rng_phi[1]-rng_phi[0]) )

    # Assign from angles to pixels for each star
    pixel = hp.ang2pix(nside, theta1, phi1)

    return pixel

def numberPixels(npixel):

    '''
    numberPixels - Find a good number of pixels to use for survey

    Parameters
    ----------
        npixel: int
            - Initial input number of pixels

    Returns
    -------
        npixel: int
            - Better number of pixels to use

    '''

    # Calculate nside from number of pixels
    nside = lambda npix: np.sqrt(npix/12)
    # float approximation to nside
    flt = nside(npixel)

    if flt == int(flt):
        # If flt is an int, valid pixel number given.
        pass
    else: # If flt is not an int, invalid pixel number given (has to be 12*nside**2)
        # Choice to use more or less pixels
        rd_down = hp.nside2npix(int(flt))
        rd_up = hp.nside2npix(int(flt)+1)

        pix = npixel
        good_pix = False
        while not good_pix:
            # Ask for another value of npixel to be given
            x = input("%d is an invalid number of pixels for healpix, can do %d or %d?" %(pix, rd_down, rd_up))

            if type(x)==int:
                pix=x
                if pix in (rd_down, rd_up): # Good number of pixels give :)
                    good_pix = True
                else: # Find roundings of pixel numbers based on given value for pix
                    rd_down = hp.nside2npix(int(nside(pix)))
                    rd_up = hp.nside2npix(int(nside(pix))+1)
            else: # npixels must be an integer
                print("Please input an integer ")
        npixel = pix

    # Tell the user how many pixels they have ended up going for
    print("%d pixels being used (nside = %d). Excellent choice :D" %(npixel, hp.npix2nside(npixel)))

    # Produce a mollweide plot of the pixels so they can see the pixellation of the sky.
    m = np.arange( npixel )
    hp.mollview(m)

    return int(npixel)