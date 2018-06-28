'''
FieldAssignment - Contains classes for assigning stars from all-sky surveys to individual fields

Classes
-------
FieldAssignment - For assigning stars from an allsky survey to the field pointings of a multi-fibre survey.

Requirements
------------

ArrayMechanics.py
'''

import numpy as np
import pandas as pd
import gc
import cProfile
import pstats

import os
import sys
import time
import gzip
from seestar import ArrayMechanics
from seestar import surveyInfoPickler
import psutil


class FieldAssignment():

    '''
    FieldAssignment - For assigning stars from an allsky survey to the field pointings of a multi-fibre survey.

    args
    ----
        fileinfo: surveyInformation instance
            - surveyInformation which includes the required format for files

    **kwargs
    --------

      photometric_files=[]: list of str
          - Paths to all files with the photometric data
          - If only one datafile, set only one element of list equal to that filename

      fieldpoint_directory = "": str
          - Name of directory where we want to store pointings files
      
      pointings_path="": str
          - Path to csv file for field pointings

      file_headers=['a', 'b', 'c', 'd', 'e']: list (length 5) of str
          - File headers in pointings file
          - File headers will be replaced with ['fieldID', 'Phi', 'Th', 'SA', 'halfangle']

      pointings_units = 'degrees': str
          - Units of angles in field pointings file.

    Extra parameters
    ----------------
      allsky_headers = []: list of str
          - Columns in allsky files which we want to use and keep.

      allsky_angles = ['glon', 'glat']
          - Column headers in allsky files which corespond to Phi and Theta angles respectively

      notnulls = ['glon', 'glat']
          - Columns which we don't want to have any nulls in when outputting the final catalogue
          - CURRENTLY UNUSED DUE TO A BUG

      allsky_total = 1
          - Expected total number of stars in allsky survey
          - Useful for getting an idea of the progress cause this can take a while

    Functions
    ---------
        ClearFiles() - Clears all field pointing files and leaves headers

        RunAssignment(): - Runs shortcut method for calculating field assignments 
        (can be substantially quicker than RunAssignmentAPTPM)

        RunAssignmentAPTPM(): - Runs thorough method for calculating field assignments

    '''

    def __init__(self, fileinfo_path,
                 photometric_files=[], 
                 fieldpoint_directory = "",
                 pointings_path="", file_headers=['a', 'b', 'c', 'd', 'e'],
                 pointings_units = 'degrees'):

        # surveyInformation instance
        self.fileinfo = surveyInfoPickler.surveyInformation(fileinfo_path)

        # List of files with full galaxy data
        # All files should be .csv format
        self.photometric_files = photometric_files

        # Total number of stars in database
        self.total = countStars(photometric_files)
        
        # Directory and names of pointings files
        self.fieldpoint_directory = self.fileinfo.photo_path
        
        # Setup pointings dataframe containing unique field positions
        pointings_path = self.fileinfo.field_path
        pointings = pd.read_csv(pointings_path)
        # Make pointings a class attribute
        self.pointings = pointings

        # ID type for fields in pointings
        self.IDtype = self.fileinfo.field_dtypes[0]

        # Number of stars to be imported at once
        self.N_import = importLimit(photometric_files)

        # Number of stars to be iterated over
        Npoint = len(pointings)
        self.N_iter = iterLimit(Npoint)

    def __call__(self):

        self.ClearFiles()
        self.RunAssignmentAPTPM(N_import=self.N_import, N_iterate=self.N_iter)

    def ClearFiles(self):

        '''
        ClearFiles() - Clears all field pointing files and leaves headers
        '''
        
        print 'Clearing field files...',
        
        # Take headers from source files (photometric catalogue)
        headers = pd.read_csv(self.photometric_files[0], nrows = 1)[:0]
        
        # Write all field files with the appropriate headers
        for field in self.pointings[self.fileinfo.field_coords[0]]:
            headers.to_csv(os.path.join(self.fieldpoint_directory, str(field))+'.csv',
                          index=False, mode = 'w')
            
        print '...done\n'
        

    def RunAssignment(self, N_import=int(1e6), N_iterate=int(1e4)):
        
        '''
        RunAssignment(): - Runs shortcut method for calculating field assignments 
        (can be substantially quicker than RunAssignmentAPTPM)

        **kwargs
        --------
        N_import = 1e6:
            - Max number of stars to import from file at any 1 time due to memory constraints. 

        N_iterate = 1000:
            - Max number of stars to assign fields at any 1 time due to memory constraints
        '''

        # Time in order to track progress
        start = time.time()
        # For analysing the progress
        starsanalysed = 0
        
        # Open files for writing
        open_files = {}
        for field in self.pointings.fieldID:
            open_files[field] = gzip.open(self.fieldpoint_directory+str(field)+'.csv', 'a+')

        # Iterate over full directory files
        for filename in self.allsky_files:

            # Number of rows in imported component
            Nfile = N_import
            # Number of rows allowed per import
            Lfile = N_import
            # Import number
            n_iter = 0
            
            
            while Lfile == Nfile:

                # Loading in full dataset
                df_allsky = pd.read_csv(self.allsky_directory+filename, 
                                         low_memory=False, compression='gzip', 
                                         usecols = self.allsky_headers,
                                         nrows = Nfile, skiprows = range(1, n_iter*Nfile))
                n_iter += 1
                Nfile = len(df_allsky)

                # Ditch any rows containing a 'null' value
                #null_condition = np.sum(np.array(df_allsky[nonulls])=='null', axis=1).astype(bool)
                #df_allsky = df_allsky[~null_condition]
                
                # Degrees to radians
                if self.allsky_units == 'degrees':
                    df_allsky[self.allsky_angles] *= np.pi/180
                    
                # Number of rows in iteration
                l=N_iterate
                # Number of rows allowed per iteration
                N=N_iterate
                # Iteration number
                i=0

                while l==N:

                    # Updates of progress continuously output
                    perc = round((starsanalysed/float(self.allsky_total))*100, 3)
                    duration = round((time.time() - start)/60., 3)
                    projected = round((time.time() - start)*self.allsky_total/((starsanalysed+1)*3600), 3)
                    sys.stdout.write('\r'+'allsky file: '+filename+'  '+\
                                   'Completion: '+str(starsanalysed)+'/'+str(self.allsky_total)+'('+
                                   str(perc)+'percent)  Time='+str(duration)+'m  Projected: '+str(projected)+'h')
                    sys.stdout.flush()
                    
                    # Take rows from df_allsky in permitted range
                    try: df = df_allsky.iloc[i*N:(i+1)*N]
                    # Index error when we reach the end of the file
                    except IndexError: df = df_allsky.iloc[i*N:]

                    # Update values with latest iteration
                    l=len(df)
                    i+=1
                    starsanalysed+=l

                    # Drop index so that indexing will work
                    df = df.reset_index(drop=True)
                    pointings = self.pointings.reset_index(drop=True)
                    pointings = pointings.copy()
                    
                    # Column header labels in pointings
                    Phi, Th, SA = ('Phi', 'Th', 'SA')
                    # Provide columns with correct headers
                    df['Phi'] = df[self.allsky_angles[0]]
                    df['Th'] = df[self.allsky_angles[1]]


                    # Shift in Phi
                    Mp_df = np.repeat([df[Phi]], len(pointings), axis=0)
                    Mp_point = np.transpose(np.repeat([pointings[Phi]],len(df), axis=0))
                    delta_p = Mp_df - Mp_point
                    delta_p[delta_p>np.pi] = 2*np.pi - (Mp_df[delta_p>np.pi] - Mp_point[delta_p>np.pi])
                    del(Mp_df, Mp_point)
                    gc.collect()
                    # Shift in Theta
                    Mt_df = np.repeat([df[Th]], len(pointings), axis=0)
                    Mt_point = np.transpose(np.repeat([pointings[Th]], len(df), axis=0))
                    delta_t = Mt_df - Mt_point
                    cos_t = np.cos(Mt_point)
                    del(Mt_df, Mt_point)
                    gc.collect()


                    # Delta is the root sum of squares (with Cos(theta) correction)
                    delta = np.sqrt(delta_t**2 + (delta_p**2)*cos_t**2)


                    Msa_point = np.transpose(np.repeat([getattr(pointings,SA)], len(df), axis=0))
                    Msa_point = np.sqrt(Msa_point/np.pi) * np.pi/180

                    Mbool = delta < Msa_point
                    
                    del(delta, Msa_point) 
                    gc.collect()
                    

                    fields = pointings.fieldID.astype(str).tolist()

                    Mbool = pd.DataFrame(np.transpose(Mbool), columns=fields)

                    for field in self.pointings.fieldID:

                        data = df[Mbool[str(field)]]

                        data[self.allsky_headers].to_csv(open_files[field], compression='gzip',
                                                          index=False, header=False)

    def RunAssignmentAPTPM(self, N_import=100000, N_iterate=1000):
            
            # Time in order to track progress
            start = time.time()
            # For analysing the progress
            starsanalysed = 0
            
            # Open files for writing
            open_files = {}
            for field in self.pointings[self.fileinfo.field_coords[0]]:
                open_files[field] = open(os.path.join(self.fieldpoint_directory, str(field))+'.csv', 'a+')

            # Iterate over full directory files
            for filename in self.photometric_files:
                
                for df_allsky in pd.read_csv(filename, chunksize=N_import):
                
                    starsanalysed += len(df_allsky)
                    fname = filename.split('/')[-1]

                    # Updates of progress continuously output
                    perc = round((starsanalysed/float(self.total))*100, 3)
                    duration = round((time.time() - start)/60., 1)
                    projected = round((time.time() - start)*self.total/((starsanalysed+1)*3600), 3)
                    sys.stdout.write('\r'+'allsky file: '+fname+'  '+\
                                   'Completion: '+str(starsanalysed)+'/'+str(self.total)+'('+
                                   str(perc)+'%)  Time='+str(duration)+'m  Projected: '+str(projected)+'h')
                    sys.stdout.flush()
                        
                    # Column header labels in pointings
                    phi, theta, halfangle = self.fileinfo.field_coords[1:4]
                        
                    df_allsky = ArrayMechanics.\
                                AnglePointsToPointingsMatrix(df_allsky, self.pointings, phi, theta, halfangle,
                                                            IDtype = self.IDtype, Nsample = N_iterate)

                    
                    for field in self.pointings[self.fileinfo.field_coords[0]]:

                        # Check which rows are assigned to the right field
                        df_bool = df_allsky.points.apply(lambda x: field in x)
                        df = df_allsky[df_bool]
                        
                        df.to_csv(open_files[field], index=False, header=False)
                        
    def RunAssignmentAPTPM_old(self, N_import=1000000, N_iterate=1000):

        '''
        RunAssignmenAPTPMt(): - Runs thorough method for calculating field assignments

        **kwargs
        --------
        N_import = 1000000:
            - Max number of stars to import from file at any 1 time due to memory constraints. 

        N_iterate = 1000:
            - Max number of stars to assign fields at any 1 time due to memory constraints
        '''
        
        # Time in order to track progress
        start = time.time()
        # For analysing the progress
        starsanalysed = 0
        
        # Open files for writing
        open_files = {}
        for field in self.pointings.fieldID:
            open_files[field] = gzip.open(self.fieldpoint_directory+str(field)+'.csv', 'a+')

        # Iterate over full directory files
        for filename in self.allsky_files:

            # Number of rows in imported component
            Nfile = N_import
            # Number of rows allowed per import
            Lfile = N_import
            # Import number
            n_iter = 0
            
            
            while Lfile == Nfile:
                
                # Updates of progress continuously output
                perc = round((starsanalysed/float(self.allsky_total))*100, 3)
                duration = round((time.time() - start)/60., 3)
                projected = round((time.time() - start)*self.allsky_total/((starsanalysed+1)*3600), 3)
                sys.stdout.write('\r'+'allsky file: '+filename+'  '+\
                               'Completion: '+str(starsanalysed)+'/'+str(self.allsky_total)+'('+
                               str(perc)+'percent)  Time='+str(duration)+'m  Projected: '+str(projected)+'h')
                sys.stdout.flush()

                # Loading in full dataset
                df_allsky = pd.read_csv(self.allsky_directory+filename, 
                                         low_memory=False, compression='gzip', 
                                         usecols = self.allsky_headers,
                                         nrows = Nfile, skiprows = range(1, n_iter*Nfile))
                n_iter += 1
                Nfile = len(df_allsky)
                starsanalysed += Nfile

                # Ditch any rows containing a 'null' value
                #print(np.array(df_allsky[self.notnulls])=='null')
                #null_condition = np.sum(np.array(df_allsky[self.notnulls])=='null', axis=1).astype(bool)
                #df_allsky = df_allsky[~null_condition]
                
                # Column header labels in pointings
                Phi, Th, SA = ('Phi', 'Th', 'SA')
                # Assume that all fields have the same solid angle
                #SA = self.pointings.SA.iloc[0]
                
                # Provide columns with correct headers
                df_allsky['Phi'] = df_allsky[self.allsky_angles[0]]
                df_allsky['Th'] = df_allsky[self.allsky_angles[1]]
                
                # Degrees to radians
                if self.allsky_units == 'degrees':
                    df_allsky[['Phi', 'Th']] *= np.pi/180
                    
                df_allsky = ArrayMechanics.\
                            AnglePointsToPointingsMatrix(df_allsky, self.pointings, Phi, Th, SA,
                                                        IDtype = self.IDtype, Nsample = N_iterate)

                
                for field in self.pointings.fieldID:

                    # Check which rows are assigned to the right field
                    df_bool = df_allsky.points.apply(lambda x: field in x)
                    df = df_allsky[df_bool]

                    df[self.allsky_headers].to_csv(open_files[field], compression='gzip',
                                                     index=False, header=False)

def importLimit(files, proportion=0.1):

    filename = files[0]
    df = pd.read_csv(filename, nrows=1000)

    mem_thousand = sys.getsizeof(df)
    mem = psutil.virtual_memory()

    ratio = mem.available/mem_thousand
    import_max = 1000 * ratio * proportion
    
    return import_max

def iterLimit(Npoint, proportion=0.1):

    mem = psutil.virtual_memory()

    iter_max = (mem.available * proportion)/(50 * Npoint)
    
    return int(iter_max)

def countStars(files):

    print "Counting total number of stars",

    count = 0
    for filen in files:
        print ".",
        with open(filen) as f:
            for _ in f:
                count += 1
    
    print "done"
    return count

