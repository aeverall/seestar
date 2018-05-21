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
import sys
import time
import gzip
import ArrayMechanics


class FieldAssignment():

    '''
    FieldAssignment - For assigning stars from an allsky survey to the field pointings of a multi-fibre survey.

    **kwargs
    --------
      allsky_directory="": str
          - Directory in which files for the allsky dataset are stored

      allsky_files=[]: list of str
          - File names in allsky dataset
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

    def __init__(self, 
                 allsky_directory="", allsky_files=[], 
                 fieldpoint_directory = "",
                 pointings_path="", file_headers=['a', 'b', 'c', 'd', 'e'],
                 pointings_units = 'degrees'):
        
        # Locations of files
        self.allsky_directory = allsky_directory
        # List of files with full galaxy data
        # All files should be .csv.gz format
        self.allsky_files = allsky_files
        self.allsky_units = 'degrees'
        # Total number of stars in database
        self.allsky_total = 1
        
        # Column headers to take from allsky
        self.allsky_headers = []
        # Column headers for angles
        self.allsky_angles = ['Phi', 'Th']
        
        # Columns which are not allowed to contain null values
        self.notnulls = ['glon', 'glat']
        
        # Directory and names of pointings files
        self.fieldpoint_directory = fieldpoint_directory
        
        # Setup pointings dataframe containing unique field positions
        pointings_path = pointings_path
        pointings = pd.read_csv(pointings_path)
        
        ### Tidying the pointings field file
        # Headers: [fieldID, glon, glat, SA, halfangle]
        use_headers = ['fieldID', 'Phi', 'Th', 'SA', 'halfangle']
        file_headers = file_headers
        pointings = pointings.rename(index=str, columns=dict(zip(file_headers, use_headers)))
        pointings = pointings.drop_duplicates(subset='fieldID')
        
        # Convert Degrees to Radians
        angle_units = pointings_units
        if angle_units == 'degrees':
            pointings.Phi *= np.pi/180
            pointings.Th *= np.pi/180
            
        # Make pointings a class attribute
        self.pointings = pointings
            
        # ID type for fields in pointings
        self.IDtype = type(self.pointings.fieldID.iloc[0])
        


    def ClearFiles(self):

        '''
        ClearFiles() - Clears all field pointing files and leaves headers
        '''
        
        print('Clearing field files...')
        
        # Take headers from source files (photometric catalogue)
        headers = pd.read_csv(self.allsky_directory+self.allsky_files[0], 
                              compression='gzip', nrows = 1)[self.allsky_headers][:0]
        
        # Write all field files with the appropriate headers
        for field in self.pointings.fieldID:
            headers.to_csv(self.fieldpoint_directory+str(field)+'.csv', compression='gzip',
                          index=False, mode = 'w')
            
        print('...done\n')
        

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
                        
    def RunAssignmentAPTPM(self, N_import=1000000, N_iterate=1000):

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