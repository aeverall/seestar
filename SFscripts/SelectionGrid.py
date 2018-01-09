'''
SelectionGrid - Contains tools for creating a selection function for a given
                datasource which uses plate based obsevation techniques

Classes
-------
    FieldInterpolants - Class for building a dictionary of interpolants for each of the survey Fields
                        The interpolants are used for creating Field selection functions

Functions
---------
    PointDataSample - Creates sample of points which correspond to a plate from RAVE catalogue

    IndexColourMagSG - Creates an interpolant of the RAVE star density in col-mag space
                     - For the points given which are from one specific observation plate

    fieldInterp - Converts grids of colour and magnitude coordinates and a 
                  colour-magnitude interpolant into the age, mh, s selection
                  function interpolant.

    plotSFInterpolants - Plots the selection function in any combination of age, mh, s coordinates.
                         Can chose what the conditions on individual plots are (3rd coord, Field...)

    findNearestFields - Returns the nearest field for each point in the given list
                        in angles (smallest angle displacement)

Requirements
------------
agama

Access to path containing Galaxy modification programes:
    sys.path.append("../FitGalMods/")

    import CoordTrans

ArrayMechanics
AngleDisks
PlotCoordinates
DataImport
'''

import numpy as np
import pandas as pd
from itertools import product
import re
import dill
import pickle

import sys, os
os.path.exists("../../Project/Milky/FitGalaxyModels/")
sys.path.append("../FitGalMods/")

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

from scipy.interpolate import RegularGridInterpolator as RGI

import CoordTrans

from ArrayMechanics import *
from AngleDisks import *
from PlotCoordinates import *
from DataImport import *

# Cleaning up:
# Recomment iterateAll in doc string
# Delete iterateTmass and iterateSurvey class functions
# Comment an explanation for how grid regions and sizes are decided
# Deal with the face that grid[grid==np.inf] = 0.0 is destroying our results


class FieldInterpolator():

    '''
    FieldInterpolants - Class for building a dictionary of interpolants for each of the survey Fields
                        The interpolants are used for creating Field selection functions

    Functions
    ---------
        StarDataframe - Creates a dataframe of relevant relevant information on stars used to create  interpolants

        PointingsDataframe - Creates a dataframe of relevant relevant information on the coordinates of RAVE Fields

        IterateFields - Runs CreateInterpolant for each Field in the RAVE database

        CreateInterpolant - Creates an interpoland in colour-magnitude space for the given rave Field
                            Adds information on the SF for the Field to self.interpolants dictionary

    Returns
    -------
        self.interpolants: dictionary of floats and interpolants
                - Dictionary with an entry for each RAVE Field with the interpolant, mag_range, col_range, grid_area
    
    Dependencies
    ------------
        EquatToGal
        
        AnglePointsToPointingsMatrix
        

    '''
    def printTest(self, string, object):
        if self.testbool:
            print(string + ': ')
            print(object)
    
    def __init__(self, survey = 'RAVE',
                 surveysf_exists = True,
                 ColMagSF_exists = True,
                 testbool = False, testfield = ''):
        
        self.testbool=testbool

        data_path = '../Data/'


        # RAVE column headers and file names:
        if survey=='RAVE':
            star_coords = ['FieldName', 'RA_TGAS', 'DE_TGAS', 'Jmag_2MASS', 'Kmag_2MASS', 'Hmag_2MASS']
            star_path = '../Data/RAVE/RAVE_wPlateIDs.csv'
            field_coords = (['RAVE_FIELD', 'RAdeg', 'DEdeg'], 'Equatorial')
            field_path = '../Data/RAVE/RAVE_FIELDS_wFieldIDs.csv'
            self.tmass_coords = ['RA', 'Dec', 'j', 'h', 'k']
            #tmass_path = '../Data/2MASS/RAVE-2MASS_sqsets/RAVEfield3_'
            tmass_path = '/media/andy/37E3-0F91/2MASS/RAVE-2MASS_fieldbyfield_29-11/RAVEfield_'

            SA = 28.3
            tmass_pickle = '../Data/RAVE/2massdata_RAVEfields.pickle'
            survey_pickle = '../Data/RAVE/surveydata_RAVEfields.pickle'
            sf_pickle = '../Data/RAVE/sf_RAVEfields.pickle'
            self.tmasstag = '.csv'
            self.fieldlabel_type = str

            if testbool:
                tmass_pickle = '../Data/RAVE/2massdata_RAVEfields_test.pickle'
                survey_pickle = '../Data/RAVE/surveydata_RAVEfields_test.pickle'
                sf_pickle = '../Data/RAVE/sf_RAVEfields_test.pickle'

        # Apogee column headers and file names:
        if survey=='Apogee':
            star_coords = ['# location_id', 'ra', 'dec', 'j', 'k', 'h']
            field_coords = (['FieldName', 'RA', 'Dec'], 'Equatorial')
            star_path = '../Data/Apogee/TGAS_APOGEE_supp_keplercannon_masses_ages.csv'
            field_path = '../Data/Apogee/apogeetgasdr14_fieldinfo.csv'
            self.tmass_coords = ['RA', 'Dec', 'j', 'h', 'k']
            #tmass_path = '../Data/Apogee/apg2mass/'
            tmass_path = '/media/andy/37E3-0F91/2MASS/Apogee-2MASS/APOGEEfield_'
            self.tmasstag = '.csv'

            SA = 7
            tmass_pickle = '../Data/Apogee/2massdata_apogeeFields_test.pickle'
            survey_pickle = '../Data/Apogee/surveydata_apogeeFields_test.pickle'
            sf_pickle = '../Data/Apogee/sf_apogeefields.pickle'

            self.fieldlabel_type = float

        #iso_pickle = "../Data/isochrone_distributions.pickle"
        iso_pickle = "../Data/Isochrones/isochrone_distributions_resampled.pickle"
        
        self.pointings = self.ImportDataframe(field_path, field_coords[0], 
                                             data_source='Fields', 
                                             angle_units='degrees',
                                             coordinates= field_coords[1],
                                             Field_solidangle=False, solidangle=SA)
        self.pointings = self.pointings.set_index('fieldID', drop=False)
        self.pointings = self.pointings.drop_duplicates(subset = 'fieldID')

        ############Testing on 1 plate at a time
        if testbool: 
            testfield = ['1758m28', '0051m27']
            self.pointings = self.pointings[self.pointings.fieldID.isin(testfield)]
        #self.printTest('List of pointings to be tested', self.pointings)
        #if survey=='RAVE': self.pointings = self.pointings[(self.pointings.RA < 235*np.pi/180)][:10]
        #                                (self.pointings.RA < 350*np.pi/180)]
        #self.pointings=self.pointings[11:14]
        print(len(self.pointings))
        print(self.pointings[['RA','Dec','l','b','half_opening']])
        ############self.pointings = self.pointings.iloc[:15]
        
        if surveysf_exists==False:

            
            # Parsec isochrones
            print("Undilling isochrone interpolants...")
            with open(iso_pickle, "rb") as input:
                self.pi = dill.load(input)
            print("...done.")
            print(" ")

            self.isoage = np.copy(self.pi['isoage']) 
            self.isomh  = np.copy(self.pi['isomh'])
    
            
            if ColMagSF_exists == False:
                
                print('Importing data for Colour-Magnitude Field interpolants...')
                self.stars = pd.DataFrame()
                self.mag_range, self.col_range = (0.,0.),(0.,0.)

                self.stars = self.ImportDataframe(star_path, star_coords, 
                                                 angle_units='degrees')
                print("...done.")
                print(" ")                
                
            
                print('Creating Colour-Magnitude Field interpolants...')
                self.tmass_interp, self.data_interp = self.iterateAllFields(tmass_path)
                print("...done.")
                print(" ")
                self.printTest('2MASS interpolants', self.tmass_interp)
                self.printTest('Survey interpolants', self.data_interp)
                
                with open(survey_pickle, 'wb') as handle:
                    pickle.dump(self.data_interp, handle)
                with open(tmass_pickle, 'wb') as handle:
                    pickle.dump(self.tmass_interp, handle)
                    
            else:
                # Once Colour Magnitude selection functions have been created
                # Unpickle colour-magnitude interpolants
                print("Unpickling colour-magnitude interpolant dictionaries...")
                with open(survey_pickle, "rb") as input:
                    self.data_interp = pickle.load(input)
                with open(tmass_pickle, "rb") as input:
                    self.tmass_interp = pickle.load(input)
                print("...done.")
                print(" ")
                
            print('Creating Distance Age Metalicity interpolants...')
            surveysf, agerng, mhrng, srng = self.createDistMhAgeInterp()
            with open(sf_pickle, 'wb') as handle:
                    pickle.dump((surveysf, agerng, mhrng, srng), handle)
            print("...done.")
            print(" ")
        
        # Once full selection function has been created
        # Unpickle survey selection function
        print("Unpickling survey selection function...")
        with open(sf_pickle, "rb") as input:
            self.surveysf, self.agerng, self.mhrng, self.srng  = pickle.load(input)
        print("...done.")
        print(" ")
        
    def __call__(self, catalogue):


        '''
        __call__ - Once the selection function has been included, this takes in a catalogue
                   of stars and returns a probability of each star in the catalogue being
                   detected given that there is a star at this point in space.

        Parameters
        ----------
            catalogue: DataFrame
                    - The catalogue of stars which we want the selection function probability of.
                    - Must contain: age, s, mh, RA, Dec

        Returns
        -------
            catalogue: DataFrame
                    - The same as the input dataframe with an additional 'sf' column

            missing_fields: list
                    - All fields which shoulld be in the database but are missing
        '''

        # Assign each point to a pointing
        catalogue = self.PointsToPointings(catalogue, Phi='RA', Th='Dec')

        # Lists which the selection function and missing fields will be added to
        listsf = []
        missing_fields = []
        print(catalogue)

        # Iterate over stars in catalogue
        for index in catalogue.index:
            # List to append values of sf for individual fields to
            sf_star = []
            # Iterate over fields corresponding to each coordinate
            for field in catalogue.points.loc[index]:
                # PointsToPointings converts all field types to strings, convert back
                field = self.fieldlabel_type(field)
                print(field)
                try:
                    interp = self.surveysf.loc[field]['agemhssf']
                    try:
                        sf = interp((catalogue.age.loc[index],
                                     catalogue.mh.loc[index],
                                     catalogue.s.loc[index]))
                    except IndexError: 
                        sf=0.
                # If field is not in surveysf, add field to missing_fields list
                except KeyError: 
                    sf=0.0
                    missing_fields.append(field)
                    print('missing_fields')
                sf_star.append(sf)

            listsf.append(np.sum(sf_star))

        catalogue['sf'] = pd.Series(listsf)

        return catalogue, missing_fields
        
        
    def ImportDataframe(self, path,
                        coord_labels, 
                        data_source = 'stars', 
                        coordinates = 'Equatorial',
                        angle_units = 'degrees',
                        Field_solidangle = True,
                        solidangle = 0.):

        '''
        ImportDataframe - Creates a dataframe of relevant relevant information on stars
                          or on field pointings.

        Parameters
        ----------
            path: string
                    - Location of the csv file which contains the database of points

            coord_labels: list of strings
                    - The labels of the coordinates: ['fieldID','RA', 'Dec']
                    - Or, if coordinates='Galactic': ['fieldID','l', 'b']

        **kwargs
        --------
            data_source: string ('stars')
                    - 'stars', assumes we are importing a database of stars
                        (so it imports magnitudes aswell)
                    - 'Fields': assumes we are importing a database of fields
                        (so it imports SolidAngle and calculates halfangle too)

            coordinates: string ('Equatorial')
                    - 'Galactic': Takes given fields to be l and b. Doesn't do a coordinate transformation
                    - 'Equatorial': Takes given fields to be RA and Dec and calculates Galactic coords

            angle_units: string ('degrees')
                    - 'degrees': Converts units to radians for the output
                    - 'rad' or 'radians': Doesnt convert units

            Field_solidangle: bool (True)
                    - True: Assumes a solid angle if given per field in the database
                    - False: Adds a column to the database which all have the same solid angle
                                (if False, solidangle= must be given)

            solidangle: float (0.)
                    - The generic solid angle extent of all plates if Field_solidangle == False

        Returns
        -------
            data: DataFrame
                    - The corrected dataframe which is in the right format for use in the rest of the class

        Contributes
        -----------
            self.mag_range: tuple
                    - Maximum and minimum values of H magnitude in the star data

            self.col_range: tuple
                    - Maximum and minimum values of J-K in the star data

        '''         

        # Use .xxx in file path to determine file type
        re_dotfile = re.compile('.+\.(?P<filetype>[a-z]+)')
        filetype = re_dotfile.match(path).group('filetype')

        # Import data as a pandas DataFrame
        data = getattr(pd, 'read_'+filetype)(path, 
                                             usecols = coord_labels)

        # Relabel columns
        if coordinates == 'Equatorial': coords = ['fieldID','RA', 'Dec']
        elif coordinates == 'Galactic': coords = ['fieldID','l', 'b']

        if data_source=='stars': coords.extend(['Jmag', 'Kmag', 'Hmag'])
        elif data_source=='Fields':
            if Field_solidangle: coords.extend(['SolidAngle'])
            else: data['SolidAngle'] = np.zeros((len(data))) + solidangle
        data = data.rename(index=str, columns=dict(zip(coord_labels, coords)))

        # Remove any null values from data
        for coord in coords: data = data[pd.notnull(data[coord])]

        # Correct units
        if (angle_units == 'degrees') & \
           (coordinates == 'Equatorial'): data.RA, data.Dec = data.RA*np.pi/180, data.Dec*np.pi/180
        elif (angle_units == 'degrees') & \
             (coordinates == 'Galactic'): data.l, data.b = data.l*np.pi/180, data.b*np.pi/180
        elif angle_units in ('rad','radians'): pass
        else: raise ValueError ("I don't understand the units specified")
        # Include Galactic Coordinates
        if coordinates == 'Equatorial': data['l'], data['b'] = EquatToGal(data.RA, data.Dec)
        if coordinates == 'Galactic': data['RA'], data['Dec'] = GalToEquat(data.l, data.b)

        if data_source=='stars': 
            data['Colour'] = data.Jmag - data.Kmag

            # Magnitude and colour ranges from full sample
            mag_range = (np.min(data.Hmag), np.max(data.Hmag))
            col_range = (np.min(data.Colour), np.max(data.Colour))

            #Save star results in the class variables
            self.mag_range = mag_range
            self.col_range = col_range

            return data

        elif data_source == 'Fields':
            #Convert solid angle to hald opening angle
            data['half_opening'] = np.sqrt(data.SolidAngle / np.pi)

            #Save Field data in the class variable
            return data
            

    def iterateAllFields(self, tmass_path):

        '''
        iterateTmassFields- Iterates over each field file of 2MASS star data and 
                            creates a colour-magnitude interpolant for the plate.

        Parameters
        ----------
            file_path: string
                    - Location of folder of 2MASS star files

        Inherited
        ---------
            self.pointings: Dataframe
                    - Dataframe of coordinates and IDs of survey fields

        Returns
        -------
            tmass_interp: dict
                    - Dictionary of interpolants with: 'interp', 'grid_area', 'mag_range', 'col_range'
                    - Doesn't contain entries for fields where there is no 2MASS plate file

        First use DataImport.reformatTmassFieldFiles to put files into correct format

        '''

        # List of fields in pointings database
        field_list = self.pointings.fieldID.values.tolist()

        tmass_interp= {}
        data_interp = {}
        
        database_coords = ['RA','DEC','J','K','H']
        df_coords = ['RA','Dec','Jmag','Kmag','Hmag']
        
        for field in field_list:
            
            # Import 2MASS data and rename magnitude columns
            tmass_points = pd.read_csv(tmass_path+str(field)+self.tmasstag)
            coords = ['Jmag', 'Hmag', 'Kmag']
            tmass_points=tmass_points.rename(index=str, 
                                             columns=dict(zip(self.tmass_coords[2:5], coords)))
            tmass_points['Colour'] = tmass_points.Jmag - tmass_points.Kmag

            # Select preferred survey stars
            survey_points = self.stars[self.stars.fieldID == field]

            # Only create an interpolant if there are any points in the region
            if len(survey_points)>0:
                # Range of colours and magnitudes
                deltamag = (np.max(tmass_points.Hmag)-np.min(tmass_points.Hmag))/np.sqrt(len(tmass_points))
                deltacol = (np.max(tmass_points.Colour)-np.min(tmass_points.Colour))/np.sqrt(len(tmass_points))

                mag_min, mag_max = (np.min(survey_points.Hmag) - deltamag/2, np.max(survey_points.Hmag) + deltamag/2)
                col_min, col_max = (np.min(survey_points.Colour) - deltacol/2, np.max(survey_points.Colour) + deltacol/2)

                # N_tmass = len(tmass_points)
                # Minimum of 3 grid boundaries in order to create a grid
                #Nside_grid = max(int(np.ceil(np.sqrt(N_tmass)) + 1), 3)
                # Set 6 grid boundaries in the interests of memory preservation
                Nside = int(np.sqrt(len(survey_points)/2))
                survey_Nside = np.max((Nside, 8))

                # Add another grid square width to the gridsize
                extra_mag = (mag_max-mag_min)/(survey_Nside-2)
                extra_col = (col_max-col_min)/(survey_Nside-2)
                mag_max += extra_mag
                mag_min -= extra_mag
                col_max += extra_col
                col_min -= extra_col

                # Number of grid boundaries
                tmass_points = tmass_points[(tmass_points.Hmag > mag_min)&\
                                            (tmass_points.Hmag < mag_max)&\
                                            (tmass_points.Colour > col_min)&\
                                            (tmass_points.Colour < col_max)]
                Nside = int(np.sqrt(len(tmass_points)/2))
                tmass_Nside = np.max((Nside, 10))
                tmass_Nside = 8
                survey_Nside = 8
                print(tmass_Nside, survey_Nside)

                # Interpolate for 2MASS data
                tmass_interpolant, tmass_gridarea, tmass_magrange, tmass_colrange = self.CreateInterpolant(tmass_points, tmass_Nside,
                                                                                                     (mag_min, mag_max), (col_min, col_max),
                                                                                                     range_limited=True)
                # Interpolate for survey data
                survey_interpolant, survey_gridarea, survey_magrange, survey_colrange = self.CreateInterpolant(survey_points, survey_Nside,
                                                                                                              (mag_min, mag_max), (col_min, col_max))

                # mag_range, col_range are now the maximum and minimum values of grid centres used in the RGI.
                tmass_interp[field] = {'interp': tmass_interpolant,
                                       'grid_area': tmass_gridarea,
                                       'mag_range': tmass_magrange,
                                       'col_range': tmass_colrange,
                                       'Nside_grid': tmass_Nside}
                # mag_range, col_range are now the maximum and minimum values of grid centres used in the RGI.
                data_interp[field] = {'interp': survey_interpolant,
                                       'grid_area': survey_gridarea,
                                       'mag_range': survey_magrange,
                                       'col_range': survey_colrange,
                                       'Nside_grid': survey_Nside,
                                       'grid_points': True}

                if (len(tmass_interp))%100 == 0: 
                    print('Number of fields interpolated: '+str(len(tmass_interp)))

            else:
                # There are no stars on the field plate
                data_interp[field] = {'grid_points': False}
            
        return tmass_interp, data_interp

    def CreateInterpolant(self, 
                          points, Nside_grid,
                          mag_range, col_range,
                          Grid = False, range_limited=False):

        '''
        CreateInterpolant - Creates an interpolant in colour-magnitude space for the given set
                            of coordinates.

        Parameters
        ----------
            points: Dataframe
                    - Set of points used to create an interpolant over col-mag space

        Dependancies
        ------------
            IndexColourMagSG - Creates an interpolant of the RAVE star density in col-mag space
                             - For the points given which are from one specific observation plate

        **kwargs
        --------
            Grid: bool (False)
                    -  if True, also returns the interpolation grid so that it can be plotted

        Returns
        -------
            interp: RGI instance
                    - Interpolant of the survey stars on the field in col-mag space

            
        if Grid... 
            pop_grid: array
                    - Interpolation grid values
            mag: array
                    - Array of values used for grid
            col: array
                    - Array of values used for grid

        else...
            grid_area: float
                    - Area per grid square in the interp grid
            mag_range: tuple
                    - Min and max of magnitudes in interpolant
            col_range: tuple
                    - Min and max of colours in interpolant
        '''

        interp, pop_grid, mag_centers, col_centers = IndexColourMagSG(points, 
                                                                     (Nside_grid, Nside_grid),
                                                                     mag_range, col_range,
                                                                     'Hmag', 'Colour',
                                                                     range_limited=range_limited)

        grid_area = (mag_centers[1] - mag_centers[0]) * (col_centers[1] - col_centers[0])
        mag_range = (np.min(mag_centers), np.max(mag_centers))
        col_range = (np.min(col_centers), np.max(col_centers))

        if Grid: return interp, pop_grid, mag, col
        else: return interp, grid_area, mag_range, col_range
        
    def sfFieldColMag(self, field, col, mag, weight):

        '''
        sfFieldColMag - Converts survey interpolant and 2MASS interpolant into a selection grid
                        in colour and magnitude space

        Parameters
        ----------
            field: string/float/int
                    - ID of the field which we're creating a selection function for

            col: array
                    - Grid of colour (J-K) values over which we're finding the value of the selection function

            mag: array
                    - Grid of H magnitude values over which we're finding the value of the selection function

        Inherited
        ---------
            self.data_interp: dict
                    - Dictionary of survey interpolants in col-mag space with col-mag ranges and grid areas given

            self.tmass_interp: dict
                    - Dictionary of 2MASS interpolants in col-mag space with col-mag ranges and grid areas given

        Returns
        -------
            grid: array
                    - Array with same dimensions as col/mag giving the ratio of survey/2MASS interpolants normalised
                      by the grid areas.
        '''

        survey = self.data_interp[field]
        tmass = self.tmass_interp[field]
        
        grid = np.zeros_like(col)
        
        # If dictionary contains nan values, the field contains no stars
        if ~np.isnan(survey['grid_area']):

            bools = (col>survey['col_range'][0])&(col<survey['col_range'][1])&\
                    (mag>survey['mag_range'][0])&(mag<survey['mag_range'][1])

            grid[bools] = (survey['interp']((mag[bools],col[bools]))/survey['grid_area'])/\
                          (tmass['interp']((mag[bools],col[bools]))/tmass['grid_area'])   
            
            grid[grid==np.inf]=0.
            grid[np.isnan(grid)] = 0.

        # Weight the grid to correspond to IMF weightings for each star
        prob = grid*weight

        return prob
            
            
    def createDistMhAgeInterp(self,
                              agemin = 0,agemax = 13,
                              mhmin=-2.5,mhmax=0.5,
                              smin=0.001,smax=20.,ns=20,
                              test=False, GoodMemory=False):

        '''
        createDistMhAgeInterp - Creates a selection function in terms of age, mh, s
                              - Integrates interpolants over isochrones.

        Inherited
        ---------
            self.data_interp: dict
                    - Dictionary of survey interpolants in col-mag space with col-mag ranges and grid areas given

            self.pointings: Dataframe
                    - Dataframe of coordinates and IDs of survey fields

            self.pi: dict
                    - Dictionary of isochrone data

            self.tmass_interp: dict
                    - Dictionary of 2MASS interpolants in col-mag space with col-mag ranges and grid areas given

            sfFieldColMag - Converts survey interpolant and 2MASS interpolant into a selection grid
                            in colour and magnitude space

        **kwargs
        --------
            agemin/agemax: float (0, 13)
                    - Age range over which we want to build the selection function

            mhmin/mhmax: float (-2.5, 0.5)
                    - mh range over which we want to build the selection function
                    
            smin/smax: float (0.001, 20.)
                    - s range over which we want to build the selection function  

            test: bool (False)
                    - If True, allows the function to be tested without running the entire class

        Returns
        -------
            fieldInfo: Dataframe
                    - Contains information on each plate and the interpolants ('agemhssf')

            (agemin,agemax): tuple of floats
                    - age range of the selection function

            (mhmin,mhmax): tuple of floats
                    - mh range of the selection function
                    
            (smin,smax): tuple of floats
                    - s range of the selection function  
        '''

        if test:
            print("Unpickling colour-magnitude interpolant dictionaries...")
            with open('../Data/surveydata_apogeeFields_test.pickle', "rb") as input:
                self.data_interp = pickle.load(input)
            with open('../Data/2massdata_apogeeFields_test.pickle', "rb") as input:
                self.tmass_interp = pickle.load(input)
            print("...done.")
            print(" ")
            data_path = '../Data/'
            # Parsec isochrones
            print("Undilling isochrone interpolants...")
            with open(data_path + "stellarprop_parsecdefault_currentmass.dill", "rb") as input:
                self.pi = dill.load(input)
            print("...done.")
            print(" ")
        
        # Copy variables locally
        fieldInfo = self.pointings
        
        isoage    = np.copy(self.pi['isoage'])
        isomh     = np.copy(self.pi['isomh'])
        isodict   = self.pi['isodict']

        # Chose the portion of isochrones used based on memory available
        if GoodMemory: # Construct age & metallicity grids with all available isochrones
            # Construct age grid
            jagemin    = max(0, 
                             np.sum(isoage<agemin)-1)
            jagemax     = min(len(isoage), 
                              np.sum(isoage<agemax) +1)
            agegrid = isoage[jagemin:jagemax]

            # Construct metallicity grid
            jmhmin     = max(0,
                             np.sum(isomh<mhmin)-1)
            jmhmax     = min(len(isomh),
                             np.sum(isomh<mhmax)+1)
            mhgrid  = isomh[jmhmin:jmhmax]
        else: # If memory on the device is limitted, this only uses a reduced set of ages / metallicities
            sizeage = 50
            sizemh = 20
            ageset = np.linspace(agemin, agemax, sizeage)
            mhset = np.linspace(mhmin, mhmax, sizemh)
            # Construct age grid
            agegrid = []
            for age in ageset:
                isochrone_age = isoage[np.abs(isoage-age) == np.min(np.abs(isoage-age))]
                agegrid.append(isochrone_age)
            agegrid = np.unique(np.array(agegrid))

            mhgrid = []
            for mh in mhset:
                isochrone_mh = isomh[np.abs(isomh-mh) == np.min(np.abs(isomh-mh))]
                mhgrid.append(isochrone_mh)
            mhgrid = np.unique(np.array(mhgrid))

        nage    = len(agegrid)
        nmh     = len(mhgrid)
              
        # Create grids for distances and selection function probabilities
        sgrid    = np.logspace(np.log10(smin),np.log10(smax),ns)
                   
        fieldInfo['tmass_bool'] = fieldInfo.fieldID.apply(lambda x: x in self.tmass_interp)
        fieldInfo=fieldInfo.set_index('fieldID', drop=False)

        # Create datagrids for selection function
        nage, nmh, ns = len(agegrid), len(mhgrid), len(sgrid)
        sfgrid  = np.zeros([nage,nmh,ns])

        # MultiIndex dataframe for applying transformations
        index = pd.MultiIndex.from_product([agegrid, mhgrid], names = ['age','mh'])
        age_mh = pd.DataFrame(list(product(agegrid, mhgrid)), columns=['age','mh'])

        # Isochrone string identifiers
        age_mh['isoname'] = "age"+age_mh.age.astype(str)+"mh"+age_mh.mh.astype(str)

        # Absolute magnitude arrays from isodict
        age_mh['absJ'] = age_mh.isoname.map(lambda x: isodict[x].Jabs)
        age_mh['absH'] = age_mh.isoname.map(lambda x: isodict[x].Habs)
        age_mh['absKs'] = age_mh.isoname.map(lambda x: isodict[x].Kabs)
        age_mh['weight'] = age_mh.isoname.map(lambda x: isodict[x].weight)

        # Grids of apparent magnitudes varying with distance over sgrid
        ns = len(sgrid)

        age_mh['isosize'] = age_mh.absJ.map(lambda x: len(x))
        fulllength = max(age_mh.isosize)
        smat = np.stack([[sgrid,]*fulllength,]*len(age_mh))

        age_mh['J_K'] = age_mh.absJ - age_mh.absKs
        age_mh[['absH', 'J_K', 'weight']] = age_mh[['absH', 'J_K', 'weight']].\
                                            applymap(lambda x: np.concatenate((x, np.zeros(fulllength-len(x)))))
                
        absHmat = np.vstack(age_mh.absH)
        J_Kmat = np.vstack(age_mh.J_K)
        weightmat = np.vstack(age_mh.weight)
                
        absHmat = np.stack([absHmat]*ns, axis=2)
        J_Kmat = np.stack([J_Kmat]*ns, axis=2)
        weightmat = np.stack([weightmat]*ns, axis=2)

        mag_conversion = 5*np.log10(smat*1000./10.)
        appHmat = mag_conversion + absHmat
        print('appHmat range of values:' + str((np.min(appHmat), np.max(appHmat))))

        age_mh = age_mh[['age', 'mh', 'isosize']]
        age_mh.set_index(['age','mh'], inplace=True)

        # Clear out unused arrays for memory
        del(smat, mag_conversion, absHmat)
        gc.collect()

        print('Undergoing field-by-field age-mh-s interpolations...')
        fieldInfo['agemhssf'] = fieldInfo.apply(self.fieldInterp, 
                                                args=(agegrid, mhgrid, sgrid, age_mh, 
                                                J_Kmat, appHmat, weightmat), axis=1)
        print('...done')

        
        return fieldInfo, (agemin, agemax), (mhmin, mhmax), (smin, smax)

    def fieldInterp(self, fieldInfo, agegrid, mhgrid, sgrid, 
                    age_mh, col, mag, weight):

        '''
        fieldInterp - Converts grids of colour and magnitude coordinates and a 
                      colour-magnitude interpolant into the age, mh, s selection
                      function interpolant.

        Parameters
        ----------
            fieldInfo: single row Dataframe
                    - fieldID and tmass_bool information on the given field

            agegrid, mhgrid, sgrid: 3D array
                    - Distribution of coordinates over which the selection function
                      will be generated

            sf: function instance from FieldInterpolator class
                    sfFieldColMag - Converts survey interpolant and 2MASS interpolant into a selection grid
                                    in colour and magnitude space

            age_mh: Dataframe
                    - Contains all unique age-metalicity values as individual rows which are then unstacked to
                      a matrix to allow for efficient calculation of the selection function.

            J_Kmat: array
                    - Matrix of colour values over the colour-magnitude space

            appJmat: array
                    - Matrix of H magnitude values over the colour-magnitude space

            weightmat: array
                    - Matrix of weightings of isochrone points so that the IMF is integrated over

        Dependencies
        ------------
            RegularGridInterpolator

        Returns
        -------
        if tmass_bool:
            sfinterpolant

        else:
            RGI([], 0)

        #### Make sure we find a better way of dealing with this so that its clear when
        #### there are problems with fields not being in the database.
        '''

        # Import field data
        fieldID = fieldInfo['fieldID']

        if self.data_interp[fieldID]['grid_points']:
            # For fields with RAVE-TGAS stars
            sys.stdout.write("\rCurrent field being interpolated: %s" % fieldID)
            sys.stdout.flush()
            

            # Selection function probabilities from colour-magnitude interpolants
            survey = self.data_interp[fieldID]
            tmass = self.tmass_interp[fieldID]

            sfprob = np.zeros_like(col)
            # If dictionary contains nan values, the field contains no stars
            if ~np.isnan(survey['grid_area']):

                # Make sure values fall within interpolant range for colour and magnitude
                # Any points outside the range will provide a 0 contribution
                bools = (col>survey['col_range'][0])&(col<survey['col_range'][1])&\
                        (mag>survey['mag_range'][0])&(mag<survey['mag_range'][1])&\
                        (col>tmass['col_range'][0])&(col<tmass['col_range'][1])&\
                        (mag>tmass['mag_range'][0])&(mag<tmass['mag_range'][1])

                sfprob[bools] = (survey['interp']((mag[bools],col[bools]))/survey['grid_area'])/\
                              (tmass['interp']((mag[bools],col[bools]))/tmass['grid_area'])   

                sfprob[sfprob==np.inf]=0.
                sfprob[np.isnan(sfprob)] = 0.
            sfprob *= weight

                           
            # Integrating (sum) selection probabilities to get in terms of age,mh,s
            integrand = np.sum(sfprob, axis=1)

            # Transform to selection grid (age,mh,s) and interpolate to get plate selection function
            age_mh['integrand'] = integrand.tolist()

            sfgrid = np.array(age_mh[['integrand']].unstack()).tolist()
            sfgrid = np.array(sfgrid)

            # Expand grids to account for central coordinates
            sfgrid, agegrid = extendGrid(sfgrid, agegrid, axis=0,
                                             x_lbound=True, x_lb=0.)
            sfgrid, mhgrid = extendGrid(sfgrid, mhgrid, axis=1)
            sfgrid, sgrid = extendGrid(sfgrid, sgrid, axis=2,
                                           x_lbound=True, x_lb=0.)

            sfinterpolant = RGI((agegrid,mhgrid,sgrid),sfgrid)

        else:
            # For fields with no RAVE-TGAS stars - 0 value everywhere in field
            print('No stars in field: '+str(fieldID))
            sfinterpolant = RGI([], 0)

        return sfinterpolant 
    
            
    def ProjectGrid(self, field):

        '''
        ProjectGrid - Produces a single interpolation of the given field

        Parameters
        ----------
            field: string/float/int
                    - Identifier for the specific rave Field which is being analysed

        Inherited
        ---------
            self.stars: Dataframe
                    - Survey stars

            self.CreateInterpolant - Creates an interpolant in colour-magnitude space for the given rave Field
                                     Adds information on the SF for the Field to self.interpolants dictionary

        Returns
        -------
            interp: RGI instance
                    - Interpolant of the survey stars on the field in col-mag space

            pop_grid: array
                    - Interpolation grid values

            mag: array
                    - Array of values used for grid

            col: array
                    - Array of values used for grid

        '''

        points = self.stars[self.stars.field_ID == field]
        
        interp, pop_grid, mag, col = self.CreateInterpolant(points, Grid=True)
        
        try:
            fig = plt.figure(figsize=(15,10))
            plt.contourf(mag, col, pop_grid)
            plt.colorbar()
            plt.xlabel('Magnitude')
            plt.ylabel('Colour')
        except ValueError:
            print(pop_grid)
            print('No stars on this Field')
        
        return interp, pop_grid, mag, col
    
    def PointsToPointings(self, stars, Phi='RA', Th='Dec'):

        '''
        PointsToPointings - Generates a list for each point for all field pointings
                            for which the point's angle coordinates fall within the
                            solid angle extent of the field.
        ### Only manages an upper limit of 20000 stars at a time!!!! ###

        Parameters
        ----------
            stars: DataFrame
                    - Points which we're attempting to assosciate with field pointings

        Inherited
        ---------
            self.pointings: Dataframe
                    - Dataframe of coordinates and IDs of survey fields

        Dependencies
        ------------
            AnglePointsToPointingsMatrix - Adds a column to the df with the number of the field pointing
                                 - Uses matrix algebra
                                    - Fastest method for asigning field pointings
                                    - Requires high memory usage to temporarily hold matrices

        Returns
        -------
            df: Dataframe
                    - stars dataset with an additional column containing lists of field IDs for each point.
        '''
        
        df = AnglePointsToPointingsMatrix(stars[:20000], self.pointings,
                                          Phi, Th, 'SolidAngle')
        
        return df

    def PlotPlates(self, EqorGal = 'Gal', **kwargs):

        '''
        PlotPlates - Plots circles for each survey field in angles

        **kwargs
        --------
            EqorGal: string ('Gal')
                    - Specifies whether to plot the plates in Galactic or Equatorial coordinates
                    - If 'Eq', plot in Equatorial coordinates

            pointings: bool (True)
                    - If True, plot all fields in self.pointings
                    - If False, plot fields which are given in fieldIDs

            fieldIDs: list
                    - list of field IDs which will be plotted if pointings=False

        Inherited
        ---------
            self.pointings: Dataframe
                    - Dataframe of coordinates and IDs of survey fields

        Dependencies
        ------------
            GenerateCircle - Creates a circle of points around a position on a sphere

            PlotDisk - Creates a Mollweide plot of the given coordinates

        Returns
        -------
            None

            Geneerates a 'mollweide' plot of circles as scatters of points.
        '''

        plates_given = {'pointings': True,
                        'fieldIDs': []}
        plates_given.update(kwargs)

        if plates_given['pointings']:
            if EqorGal == 'Eq':
                PhiRad = np.copy(self.pointings.RA)
                ThRad = np.copy(self.pointings.Dec)
            elif EqorGal == 'Gal':
                PhiRad = np.copy(self.pointings.l)
                ThRad = np.copy(self.pointings.b)
            else:
                raise ValueError('EqorGal must be either "Eq" or "Gal"!')       
        else:
            if EqorGal == 'Eq':
                PhiRad = np.copy(self.pointings.loc[fieldIDs].RA)
                ThRad = np.copy(self.pointings.loc[fieldIDs].Dec)
            elif EqorGal == 'Gal':
                PhiRad = np.copy(self.pointings.loc[fieldIDs].l)
                ThRad = np.copy(self.pointings.loc[fieldIDs].b)
            else:
                raise ValueError('EqorGal must be either "Eq" or "Gal"!')  

        SA = np.copy(self.pointings.SolidAngle)

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='mollweide')

        for i in range(len(PhiRad)):
            Phi, Th = GenerateCircle(PhiRad[i],ThRad[i],SA[i])
            PlotDisk(Phi, Th, ax)
    


def PointDataSample(df, pointings, plate):

    '''
    PointDataSample - Creates sample of points which correspond to a plate from RAVE catalogue

    Parameters
    ----------
        df: DataFrame
                - Data from the RAVE catalogue with a 'plate' field

        pointings: DataFrame
                - Dataframe of plate coordinates from Wojno's work

        plate: string
                - Id of plate

    Returns
    -------
        points: DataFrame
                - Data about stars on plate including the Theta, Phi coordinates from z-axis

        plate_coord: tuple of floats (radians)
                - Galactic coordinates of centre of the observation plate

    '''

    points = df[df.plate == plate]
    plate_coords = pointings[pointings.plate == plate]
    plate_coords['RArad'], plate_coords['DErad'] = plate_coords.RAdeg*np.pi/180, plate_coords.DEdeg*np.pi/180
    plate_coords['l'], plate_coords['b'] = EquatToGal(plate_coords.RArad, plate_coords.DErad)

    lc, bc = plate_coords.l.iloc[0], plate_coords.b.iloc[0]
    points['Phi'], points['Theta'] = InverseRotation(points.l, points.b, lc, bc)
    points['Th_zero'] = AngleShift(points.Theta)
    plate_coord = (lc,bc)

    points['J_K'] = points.Jmag_2MASS - points.Kmag_2MASS

    return points, plate_coord


def IndexColourMagSG(points, N_2D, 
                     mag_range, col_range,
                     mag_label, col_label,
                     range_limited=False):

    '''
    IndexColourMagSG - Creates an interpolant of the RAVE star density in col-mag space
                     - For the points given which are from one specific observation plate

    Parameters
    ----------
        points: DataFrame
                - data on points from one observation plate (including magnitudes and colours)

        N_2D: tuple of ints
                - (Nmags, Ncols) number of grid boundaries used to disperse data into a col-mag grid
                - NB this is #boundaries = #gridsections + 1

    **kwargs
    --------
        GoodRange: bool - True
                - Specifies whether the range of colours and magnitudes is large enough to use to specify grid
                - If False, col/mag range from the entire database is used to define the range

        mag_range: tuple of floats - (0.,0.)
                - Defines the range of magnitudes in the grid if GoodRange=False

        col_range: tuple of floats - (0.,0.)
                - Defines the range of colours in the grid if GoodRange=False

    Returns
    -------
        interpolation: scipy.interpolate.RegularGridInterpolator object
                - Function which returns the interpolated star number density at a point is given coordinates

        selection_grid: array of ints
                - Colour Magnitude grid with number of stars per grid square as the entries

        mag_centers: 1D array of floats
                - Centre coordinates of grid sections in selection_grid

        col_centers: 1D array of floats
                - Centre coordinates of grid sections in selection_grid
    '''

    Nmags = N_2D[0]
    Ncolours = N_2D[1]

    # Make copy of points in order to save gc.collect runtime
    points = pd.DataFrame(np.copy(points), columns=list(points))

    magnitudes = np.linspace(mag_range[0], mag_range[1], Nmags)
    colours = np.linspace(col_range[0], col_range[1], Ncolours)


    # Assign points to the correct index coordinates
    mag_mintomax = (magnitudes[Nmags-1] - magnitudes[0])
    col_mintomax = (colours[Ncolours-1] - colours[0])

    mag_values = (Nmags-1) * (getattr(points,mag_label) - magnitudes[0])/mag_mintomax
    col_values = (Ncolours-1) * (getattr(points,col_label)- colours[0])/col_mintomax
    mag_index = np.floor(np.float64(mag_values))
    col_index = np.floor(np.float64(col_values))
    points['Mag_index'] = mag_index.astype(int)
    points['Col_index'] = col_index.astype(int)
    points['count'] = np.zeros(len(points)) # used for labelling later
        
    # Create a dataframe of counts per coordinate
    counts = points.groupby(['Col_index', 'Mag_index']).count()

    # Create a template dataframe which governs the shape of the selection grid
    iterables = [np.arange(0,Ncolours-1,1), np.arange(0,Nmags-1,1)]
    index = pd.MultiIndex.from_product(iterables, names = ['Col_index', 'Mag_index'])
    template = pd.DataFrame(np.zeros(((Nmags-1)*(Ncolours-1),1)), index=index)

    # Merge counts and template together
    counts = template.merge(counts, how='outer', right_index=True, left_index=True)['count']

    # Unstack counts to create the selection grid (nan values - 0)
    selection_grid = counts.unstack()
    selection_grid[pd.isnull(selection_grid)]=0.
    selection_grid = np.array(selection_grid).astype(int)


    # Find grid square centre coordinates
    mag_centers = BoundaryToCentre(magnitudes)
    col_centers = BoundaryToCentre(colours)

    # Extend grids to deal with boundary effects
    # For 2MASS, range is limited by RAVE range therefore we need to extend non-zero values out

    if range_limited:

        dmag = mag_centers[1]-mag_centers[0]
        mag_lb, mag_ub = mag_centers[0]-dmag, mag_centers[len(mag_centers)-1]+dmag
        dcol = col_centers[1]-col_centers[0]
        col_lb, col_ub = col_centers[0]-dcol, col_centers[len(col_centers)-1]+dcol
        selection_grid, mag_centers = extendGrid(selection_grid, mag_centers, axis=0,
                                                x_lbound=True, x_lb=mag_lb,
                                                x_ubound=True, x_ub=mag_ub)
        selection_grid, col_centers = extendGrid(selection_grid, col_centers, axis=1,
                                                x_lbound=True, x_lb=col_lb,
                                                x_ubound=True, x_ub=col_ub)
    # For RAVE, region is limited by population therefore we need to extend zeros out
    else:
        selection_grid, mag_centers = extendGrid(selection_grid,
                                                mag_centers, axis=0)
        selection_grid, col_centers = extendGrid(selection_grid,
                                                col_centers, axis=1)


    # Use scipy.RegularGridInterpolator to interpolate
    interpolation = RGI((mag_centers, col_centers), np.transpose(selection_grid))

    return interpolation, selection_grid, mag_centers, col_centers


def plotSFInterpolants(fieldInts, (varx, vary), var1, var2, 
                        srange = (0.01, 20.), 
                        continuous=False, nlevels=10, title='',
                        save=False, fname='', **kwargs):

    '''
    plotSFInterpolants - Plots the selection function in any combination of age, mh, s coordinates.
                         Can chose what the conditions on individual plots are (3rd coord, Field...)

    Parameters
    ----------
        fieldInts: DataFrame
                - Database of selection function interpolants for each field in the survey

        (varx, vary): tuple of strings
                - Must be any combination of 'age', 'mh', 's'
                - Determines the variables for the x and y axes of the plots respectively

        var1: string
                - Variable which changes over different columns of plots
                - 'age', 'mh', 's', 'fields', 'l', 'b'

        var2: string
                - Variable which changes over different rows of plots
                - 'age', 'mh', 's', 'fields', 'l', 'b'

    **kwargs
    --------
        age, mh, s: 1D array
                - arrays of values which the coordinates will vary over

        fields: list
                - field IDs which the plots will use selection functions from

        Phi, Th: 1D array
                - If var1, var2 == 'l', 'b' - the positions of plates will vary over this range
                - My not be exactly right positions as this depends on the plate position

        pointings: list
                - Database of field pointings - only contains pointings where a selection function
                  interpolant has correctly been produced (tmass_bool==True)

    Dependencies
    ------------
        findNearestFields - Returns the nearest field for each point in the given list
                            in angles (smallest angle displacement)

    Returns
    -------
        None

        Plots an axis-shared multi-plot of the specified fields/coordinates selection functions.
    '''

    # Array for the coordinates in each dimension
    smin, smax, ns        = srange[0], srange[1], 30
    agemin, agemax, nage      = 0.0001, 13, 50
    mhmin, mhmax, nmh       = -2.15, 0.4, 25
    
    smod    = np.logspace(np.log10(smin),np.log10(smax),ns)
    agemod  = np.linspace(agemin,agemax,nage)
    agemod    = np.logspace(np.log10(agemin),np.log10(agemax),nage)
    print(agemod)
    mhmod   = np.linspace(mhmin,mhmax,nmh)
    fields = [4120.0]
    Phi = np.linspace(0, 2*np.pi, 5)
    Th = np.linspace(-np.pi/2, np.pi/2, 5)
    
    options = {'age': agemod,
               'mh': mhmod,
               's': smod,
               'fields': fields,
               'Phi': Phi,
               'Th': Th,
               'pointings': []}
    options.update(kwargs)
    
    agemod = options['age']
    mhmod = options['mh']
    smod = options['s']
    
    # Create list of fields which correspond to given array coordinates
    if (var1, var2) == ('l', 'b'):
        pointings = options['pointings']
        Phi = np.array(options['Phi']).repeat(len(options['Th']))
        Th = np.tile(options['Th'], len(options['Phi']))
        fields = findNearestFields((Phi,Th), pointings, var1, var2)
    else: fields = options['fields']


    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod, 'mh': mhmod,
                  'fields': fields, 'l': options['Phi'], 'b': options['Th']}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]",
                  'fields': 'Field ID'}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s3d = np.transpose(np.stack([[smod,]*len(agemod)]*len(mhmod)), (1,0,2))
    age3d = np.transpose(np.stack([[agemod,]*len(smod)]*len(mhmod)), (2,0,1))
    mh3d = np.transpose(np.stack([[mhmod,]*len(smod)]*len(agemod)), (0,2,1))
    
    # Transposes grids to [x, y, z] = [varx, vary, varz]
    coordinates = {'age':0,'mh':1,'s':2,'fields':3}
    a, b =coordinates[varx], coordinates[vary]
    c = 3-(a+b)
    sf3d = {}
    for fieldID in fields:
        grid = fieldInts.agemhssf.loc[fieldID]((age3d, mh3d, s3d))
        sf3d[fieldID] = grid.transpose((a,b,c))

    # Normalise contour levels to see the full range of contours
    gridmax = np.max([sf3d[x] for x in sf3d])
    levels = np.linspace(0, gridmax, nlevels)
    #levels = np.linspace(0, 1, 10)
    print(gridmax)
    
    # Set up figure with correct number of plots
    nxplots = len(axis_ticks[var1])
    nyplots = len(axis_ticks[var2])
    fig = plt.figure(figsize=(nxplots*8, nyplots*8))
    gs1 = gridspec.GridSpec(nyplots, nxplots)
    gs1.update(wspace=0.0, hspace=0.0) 
    # Dictionary to hold subplot instances so that they can be indexed
    plotdict = {}

    for index in range(nxplots*nyplots):
        x = index%nxplots
        y = index/nxplots
        #Correct gs_index so that we're counting from bottom left, not top left
        gs_index = index+ nxplots*(nyplots-1-2*(index/nxplots))
        
        # For the coordinate systems being plotted, generate grids and plot labels
        if var1 == 'fields':
            # Therefore var2 is age, mh or s
            selgrid = sf3d[fields[x]][:,:,y]
            v1text = 'Field =  '+str(fields[x])
            v2text = axis_labels[var2]+' =  '+str(options[var2][y])
            figtext = v1text+'\n'+v2text
        elif var1 in ('age', 'mh', 's'):
            # Therefore var2 is field
            selgrid = sf3d[fields[y]][:,:,x]
            v1text = axis_labels[var1]+' =  '+str(options[var1][x])
            v2text = 'Field =  '+str(fields[y])
            figtext = v1text+'\n'+v2text
        elif (var1, var2) == ('l','b'):
            selgrid = sf3d[fields[index]][:,:,0]
            l = pointings.loc[fields[index]].l
            b = pointings.loc[fields[index]].b
            v1text = 'l =  '+str.format('{0:.1f}',l*180/np.pi)
            v2text = 'b =  '+str.format('{0:.1f}',b*180/np.pi)
            fieldtext = 'Field =  '+str(fields[index])
            # Print value of static variable
            static = list(set(('age','s','mh'))-set((varx,vary)))[0]
            vtext = axis_labels[static]+' = '+str.format('{0:.1f}',options[static][0])
            figtext = v1text+'\n'+v2text+'\n'+fieldtext+'\n'+vtext
        
        # Contourf takes the transpose of the matrix
        selgrid = np.transpose(selgrid)
        # Normalise the plot so that features are still clearly seen compared with others
        #selgrid = selgrid/selgrid.max()
        
        # List which will contain the ticklabels which will be made invisible on the plot
        ticklabels = []
        
        # If index==0 - both x and y axes should be visible
        if index==0: 
            plotdict[0] = plt.subplot(gs1[gs_index])
            plt.xlabel(axis_labels[varx], **Afont)
            plt.ylabel(axis_labels[vary], **Afont)
        # If only x==0 - y-axis should be shared with index:0
        elif x==0: 
            plotdict[index] = plt.subplot(gs1[gs_index], sharex=plotdict[0])
            ticklabels = plotdict[index].get_xticklabels()
            plt.ylabel(axis_labels[vary], **Afont)
        # If only y==0 - x-axis should be shared with index:0
        elif y==0: 
            plotdict[index] = plt.subplot(gs1[gs_index], sharey=plotdict[0])
            ticklabels = plotdict[index].get_yticklabels()
            plt.xlabel(axis_labels[varx], **Afont)
        # If neither x,y==0 - both axes should be shared with x=0,y[0 plots]
        else:
            sharex = plotdict[x]
            sharey = plotdict[y*nxplots]
            plotdict[index] = plt.subplot(gs1[gs_index], sharex=sharex,
                                                         sharey=sharey)
            ticklabels = plotdict[index].get_yticklabels() + plotdict[index].get_xticklabels() 
        
        # Plot the grid on the plotting instance at the given index
        ax = plotdict[index]

        if continuous:
            im = ax.imshow(selgrid, vmin = levels.min(), vmax = levels.max(),
                             extent = [axis_ticks.get(varx).min(), axis_ticks.get(varx).max(),
                                       axis_ticks.get(vary).min(), axis_ticks.get(vary).max()])  
        else:  
            im = ax.contourf(axis_ticks.get(varx),axis_ticks.get(vary), selgrid,
                             levels=levels,colormap='YlGnBu')    
        # Add text to the plot to show the position of the field and other information
        ax.annotate(figtext, xy=(ax.get_xlim()[1]*0.1, ax.get_ylim()[0]*2), 
                    color='orange', **Afont)
        # Distance scale is set to log space
        if varx == 's': ax.set_xscale('log')
        if vary == 's': ax.set_yscale('log')
        #if varx == 'age': ax.set_xscale('log')
        # Make any labels on plots sharing axes invisible to make the plots more clear
        plt.setp(ticklabels, visible=False)

    fig.suptitle(title)
    # Add a colour bar to show the variation over the seleciton function.
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if save:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


def findNearestFields(anglelist, pointings, Phistr, Thstr):

    '''
    findNearestFields - Returns the nearest field for each point in the given list
                        in angles (smallest angle displacement)

    Parameters
    ----------
        anglelist: tuple of arrays
                - First array in tuple is Phi coordinates, second is Th coordinates
                - Angle coordinates of points in question.

        pointings: DataFrame
                - Database of field pointings which we're trying to match to the coordinates

        Phistr: stirng
                - Phi coordinate ( 'RA' of 'l' )

        Thstr: stirng
                - Th coordinate ( 'Dec' of 'b' )

    Dependencies
    ------------
        AngleSeparation - returns the angular seperation between 2 points

    Returns
    -------
        fieldlist - list of fields which coorrespont to closest pointings to the points.
    '''
    
    fieldlist = []
    
    for i in range(len(anglelist[0])):
        Phi = anglelist[0][i]
        Th = anglelist[1][i]
        points = pointings[[Phistr, Thstr, 'fieldID']]
        displacement = AngleSeparation(points[Phistr], points[Thstr], Phi, Th)
        field = points[displacement == displacement.min()]['fieldID'].iloc[0]
        
        fieldlist.append(field)
        
    return fieldlist

def plot1DSF(fieldInts, var, scale = 'lin', **kwargs):

    # Array for the coordinates in each dimension
    smin, smax, ns        = 0.01, 20., 50
    agemin, agemax, nage      = 1, 13, 50
    mhmin, mhmax, nmh       = -2.15, 0.4, 50
    
    if scale == 'log': smod = np.logspace(np.log10(smin),np.log10(smax),ns)
    else: smod = np.linspace(smin,smax,ns)
    agemod  = np.linspace(agemin,agemax,nage)
    mhmod   = np.linspace(mhmin,mhmax,nmh)
    fieldID = 4120.0
    Phi = np.linspace(0, 2*np.pi, 5)
    Th = np.linspace(-np.pi/2, np.pi/2, 5)
    
    options = {'age': agemod,
               'mh': mhmod,
               's': smod,
               'fieldID': fieldID,
               'Phi': Phi,
               'Th': Th,
               'pointings': []}
    options.update(kwargs)
    
    agemod = options['age']
    mhmod = options['mh']
    smod = options['s']


    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod, 'mh': mhmod,
                  'fieldID': fieldID, 'l': options['Phi'], 'b': options['Th']}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]",
                  'fieldID': 'Field ID'}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s3d = np.transpose(np.stack([[smod,]*len(agemod)]*len(mhmod)), (1,0,2))
    age3d = np.transpose(np.stack([[agemod,]*len(smod)]*len(mhmod)), (2,0,1))
    mh3d = np.transpose(np.stack([[mhmod,]*len(smod)]*len(agemod)), (0,2,1))

    sf3d = fieldInts.agemhssf.loc[fieldID]((age3d, mh3d, s3d))
        
    # Transposes grids to [x, y, z] = [varx, vary, varz]
    coordinates = {'age':0,'mh':1,'s':2}
    a=coordinates[var]
    if a==1: b=2
    else: b = 3-(a+1)
    c = 3-(a+b)
    sf3d = sf3d.transpose(b,c,a)
    
    # Set up figure with correct number of plots
    fig = plt.figure(figsize=(10,10))

    plt.plot(axis_ticks[var], sf3d[0,0,:])
    if scale == 'log': plt.xscale('log')
    plt.xlabel(axis_labels[var])
    
    return sf3d[0,0,:]

def plotSumSF(fieldInts, var, scale = 'lin', **kwargs):

    # Array for the coordinates in each dimension
    smin, smax, ns        = 0.001, 20., 50
    agemin, agemax, nage      = 1, 13, 50
    mhmin, mhmax, nmh       = -2.15, 0.4, 50
    
    if scale == 'log': smod = np.logspace(np.log10(smin),np.log10(smax),ns)
    else: smod = np.linspace(smin,smax,ns)
    agemod  = np.linspace(agemin,agemax,nage)
    mhmod   = np.linspace(mhmin,mhmax,nmh)
    fieldID = 4120.0
    Phi = np.linspace(0, 2*np.pi, 5)
    Th = np.linspace(-np.pi/2, np.pi/2, 5)
    
    options = {'age': agemod,
               'mh': mhmod,
               's': smod,
               'fieldID': fieldID,
               'Phi': Phi,
               'Th': Th,
               'pointings': []}
    options.update(kwargs)
    
    agemod = options['age']
    mhmod = options['mh']
    smod = options['s']


    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod, 'mh': mhmod,
                  'fieldID': fieldID, 'l': options['Phi'], 'b': options['Th']}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]",
                  'fieldID': 'Field ID'}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s3d = np.transpose(np.stack([[smod,]*len(agemod)]*len(mhmod)), (1,0,2))
    age3d = np.transpose(np.stack([[agemod,]*len(smod)]*len(mhmod)), (2,0,1))
    mh3d = np.transpose(np.stack([[mhmod,]*len(smod)]*len(agemod)), (0,2,1))

    sf3d = fieldInts.agemhssf.loc[fieldID]((age3d, mh3d, s3d))
        
    # Transposes grids to [x, y, z] = [varx, vary, varz]
    coordinates = {'age':0,'mh':1,'s':2}
    a=coordinates[var]
    if a==1: b=2
    else: b = 3-(a+1)
    c = 3-(a+b)
    sf3d = sf3d.transpose(b,c,a)
    
    sf = sf3d.sum(axis=0).sum(axis=0)
    
    
    # Set up figure with correct number of plots
    fig = plt.figure(figsize=(10,10))

    plt.plot(axis_ticks[var], sf)
    if scale == 'log': plt.xscale('log')
    plt.xlabel(axis_labels[var])
    
    return sf


def plotColMagInterpolants(interp, compare=False, interp2='', 
                        save=False, saven='', title='', **kwargs):
    if compare:    
        options = {'col_range': (max(interp['col_range'][0], interp2['col_range'][0]),
                                 (min(interp['col_range'][1], interp2['col_range'][1]))),
                   'mag_range': (max(interp['mag_range'][0], interp2['mag_range'][0]),
                                 (min(interp['mag_range'][1], interp2['mag_range'][1])))}
    else:
        options = {'col_range': (interp['col_range'][0], interp['col_range'][1]),
                   'mag_range': (interp['mag_range'][0], interp['mag_range'][1])}
    options.update(kwargs)
    
    # Array for the coordinates in each dimension
    colmin, colmax, ncol  = options['col_range'][0], options['col_range'][1], 30
    magmin, magmax, nmag  = options['mag_range'][0], options['mag_range'][1], 50
    
    colmod  = np.linspace(colmin+1e-4,colmax-1e-4,ncol)
    magmod   = np.linspace(magmin+1e-4,magmax-1e-4,nmag)
    
    options = {'col': colmod,
               'mag': magmod,
               'pointings': []}
    options.update(kwargs)
    
    colmod = options['col']
    magmod = options['mag']

    # Labels and ticks for the grid plots
    axis_ticks = {'col': colmod, 'mag': magmod}
    axis_labels = {'col': r"J - K",
                  'mag': r"H"}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over col, mag, s
    col2d = np.transpose(np.stack([colmod,]*len(magmod)), (1,0))
    mag2d = np.transpose(np.stack([magmod,]*len(colmod)), (0,1))
    
    grid = interp['interp']((mag2d,col2d))/interp['grid_area']
    if compare:
        grid2 = interp2['interp']((mag2d,col2d))
        #grid2[grid2<10] = 0.
        grid2 = grid2/interp2['grid_area']
        grid = grid/grid2

    # Set up figure with correct number of plots
    fig = plt.figure(figsize=(10,10))

        
    # Contourf takes the transpose of the matrix
    grid = np.transpose(grid)

        
    # Plot the grid on the plotting instance at the given inde
    im = plt.contourf(axis_ticks.get('col'),axis_ticks.get('mag'), grid,
                     colormap='YlGnBu')
    plt.xlabel(r'$J-K$')
    plt.ylabel(r'$m_H$')
    plt.title(title)

    # Add a colour bar to show the variation over the seleciton function.
    fig.colorbar(im)

    # If the fig can be saved
    if save: fig.savefig(saven, bbox_inches='tight')