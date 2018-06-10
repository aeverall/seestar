'''
SelectionGrid - Contains tools for creating a selection function for a given
                SFscopic source which uses plate based obsevation techniques

Classes
-------
    SFGenerator - Class for building a dictionary of interpolants for each of the survey Fields
                        The interpolants are used for creating Field selection functions

    obsMultiFunction - 

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
import re, dill, pickle, multiprocessing
import cProfile, pstats
import sys, os
from scipy.interpolate import RegularGridInterpolator as RGI

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

from ArrayMechanics import *
import StatisticalModels
import FieldUnions
import SFInstanceClasses
import IsochroneScaling
import surveyInfoPickler


class SFGenerator():

    '''
    FieldInterpolants - Class for building a dictionary of interpolants for each of the survey Fields
                        The interpolants are used for creating Field selection functions

    Parameters
    ----------
        pickleFile - str - path to pickled instance of class with file names and coordinate lists

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
    
    def __init__(self, pickleFile,
                 surveysf_exists = True,
                 ColMagSF_exists = True,
                 isointerp_exists = True,
                 overlapdata_exists = False):

        # Change message to check that SF is running for updated version
        print("Latest Version")

        # Load survey information as a blank class (for loading in data.)
        file_info = surveyInfoPickler.surveyInformation()
        # Get file names and coordinates from pickled file
        file_info.load(pickleFile)

        spectro_coords = file_info.spectro_coords
        spectro_path = file_info.spectro_path

        self.photo_coords = file_info.photo_coords
        self.photo_path = file_info.photo_path

        self.field_coords = file_info.field_coords
        field_path = file_info.field_path

        obsSF_pickle_path = file_info.obsSF_pickle_path
        sf_pickle_path = file_info.sf_pickle_path

        SA = file_info.field_SA

        self.photo_tag = file_info.photo_tag
        self.fieldlabel_type = file_info.fieldlabel_type

        self.iso_pickle = file_info.iso_pickle_path
        self.isocolmag_pickle = file_info.iso_interp_path
        self.isointerp_exists = isointerp_exists

        # Import the dataframe of pointings        
        pointings = self.ImportDataframe(field_path, self.field_coords[0], 
                                         data_source='Fields', 
                                         angle_units='degrees',
                                         coordinates= self.field_coords[1],
                                         Field_solidangle=False, solidangle=SA)
        pointings = pointings.set_index('fieldID', drop=False)
        pointings = pointings.drop_duplicates(subset = 'fieldID')
        #pointings = pointings[pointings.fieldID==2001.0]
        self.pointings = pointings
        

        # surveysf_exists true if Selection Function has already been calculated
        if not surveysf_exists:

            print('Creating Distance Age Metalicity interpolants...')
             #surveysf, agerng, mhrng, srng = self.createDistMhAgeInterp()
            instanceSF, instanceIMFSF, agerng, mhrng, magrng, colrng = self.createDistMhAgeInterp()
            # Comvert classes to dictionaries of attributes
            instanceSF_dict = vars(instanceSF)
            instanceIMFSF_dict = vars(instanceIMFSF)
            with open(sf_pickle_path, 'wb') as handle:
                    pickle.dump((instanceSF_dict, instanceIMFSF_dict, agerng, mhrng, magrng, colrng), handle)
            self.instanceSF=instanceSF
            self.instanceIMFSF=instanceIMFSF
            self.cm_limits = (magrng[0], magrng[1], colrng[0], colrng[1])
            print("...done.\n")
        else:              
            # Once full selection function has been created
            # Unpickle survey selection function
            print("Unpickling survey selection function...")
            with open(sf_pickle_path, "rb") as input:
                instanceSF_dict, instanceIMFSF_dict, self.agerng, self.mhrng, \
                magrng, colrng = pickle.load(input)
            print("...done.\n") 

            # Load class instance from saved dictionary
            self.instanceSF = SFInstanceClasses.intMassSF_isocalc()
            self.instanceIMFSF = SFInstanceClasses.intSF_isocalc()
            SFInstanceClasses.setattrs(self.instanceSF, **instanceSF_dict)
            SFInstanceClasses.setattrs(self.instanceIMFSF, **instanceIMFSF_dict)

            self.cm_limits = (magrng[0], magrng[1], colrng[0], colrng[1])


        # ColMagSF_exists true if Observable coord SF has already been calculated
        if not ColMagSF_exists:
            
            print('Importing data for Colour-Magnitude Field interpolants...')
            self.stars = self.ImportDataframe(spectro_path, spectro_coords, 
                                             angle_units='degrees')
            print("...done.\n")
        
            print('Creating Colour-Magnitude Field interpolants...')
            obsSF_dicts = self.iterateAllFields()
            print('\nnow pickling them...')
            with open(obsSF_pickle_path, 'wb') as handle:
                pickle.dump(obsSF_dicts, handle)
            print("...done\n.")
        else:
            # Once Colour Magnitude selection functions have been created
            # Unpickle colour-magnitude interpolants
            print("Unpickling colour-magnitude interpolant dictionaries...")
            with open(obsSF_pickle_path, "rb") as input:
                obsSF_dicts = pickle.load(input)
            print("...done.\n")

        # Load classes from dictionaries
        # Initialise dictionary
        self.obsSF = {}
        for field in obsSF_dicts:
            # Initialise class instance
            obsSF_field = SFInstanceClasses.observableSF(field)
            # Set class attributes from dictionary
            SFInstanceClasses.setattrs(obsSF_field, **obsSF_dicts[field])
            # Add class instance to dictionary
            self.obsSF[field] = obsSF_field

        # Cannot construct overlapping field system if only one field
        # This selection function doesn't currently work for a single field
        if len(self.pointings)>1:
            if not overlapdata_exists:
                # Create the field intersection database
                database = FieldUnions.CreateIntersectionDatabase(5000, pointings, self.fieldlabel_type)
                database.to_csv(file_info.overlap_path)
            else: database = pd.read_csv(file_info.overlap_path)
            self.FUInstance = FieldUnions.FieldUnion(database)
        else: raise ValueError('Cannot currently run this code with only one field, working on improving this!')
        
    def __call__(self, catalogue, method='intrinsic', 
                coords = ['age', 'mh', 's', 'mass'], 
                angle_coords=['l','b'], angles='Galactic'):

        '''
        __call__ - Once the selection function has been included, this takes in a catalogue
                   of stars and returns a probability of each star in the catalogue being
                   detected given that there is a star at this point in space.

        Parameters
        ----------
            catalogue: DataFrame
                    - The catalogue of stars which we want the selection function probability of.
                    - Must contain: age, s, mh, l, b

        Returns
        -------
            catalogue: DataFrame
                    - The same as the input dataframe with an additional columns:
                    - 'inion', 'field_info', 'points'
        '''

        if method=='observable': 
            SFcalc = lambda field, df: np.array( self.obsSF[field]((df[coords[0]], df[coords[1]])) )
        elif method=='IMFint':
            SFcalc = lambda field, df: self.instanceIMFSF( (df[coords[0]], df[coords[1]], df[coords[2]]), self.obsSF[field] )
        elif method=='int':
            SFcalc = lambda field, df: self.instanceSF( (df[coords[0]], df[coords[1]], df[coords[3]], df[coords[2]]), self.obsSF[field] )
        else: raise ValueError('Method is unknown')

        if self.field_coords[1] == 'Galactic':point_coords=['l', 'b']
        elif self.field_coords[1] == 'Equatorial':point_coords=['RA', 'Dec']
        else: raise ValueError("MThe entry in surveyInfo of field_coords should be Galactic or Equatorial, it's currently %s" % self.field_coords[1])

        #print(SFcalc(2.0, catalogue[['Happ', 'Colour', 's', 'age', 'mh', 'mass']]))

        # catalogue[points] - list of pointings which coordinates lie on
        # catalogue[field_info] - list of tuples: (P(S|v), field)
        print('Calculating all SF values...')
        catalogue = FieldUnions.GenerateMatrices(catalogue, self.pointings, angle_coords, point_coords, 'SolidAngle', SFcalc)
        print('...done')
        # The SF probabilities and coordinates of overlapping fields are used to calculate
        # the field union.
        print('Calculating union contribution...')
        catalogue['union'] = catalogue.field_info.map(self.FUInstance.fieldUnion)
        print('...done')

        return catalogue
        
        
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

        if data_source=='stars': coords.extend(['appA', 'appB', 'appC'])

        elif data_source=='Fields':
            coords.extend(['mlBound', 'muBound', 'clBound', 'cuBound'])
            if Field_solidangle: coords.extend(['SolidAngle'])
            else: data['SolidAngle'] = np.zeros((len(data))) + solidangle

        print(dict(zip(coord_labels, coords)))
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
            data['Colour'] = data.appA - data.appB

            # Magnitude and colour ranges from full sample
            mag_range = (np.min(data.appC), np.max(data.appC))
            col_range = (np.min(data.Colour), np.max(data.Colour))

            #Save star results in the class variables
            self.mag_range = mag_range
            self.col_range = col_range

            print(mag_range, col_range)

            return data

        elif data_source == 'Fields':
            #Convert solid angle to hald opening angle
            data['half_opening'] = np.sqrt(data.SolidAngle / np.pi)

            #Save Field data in the class variable
            return data
            
    def iterateAllFields(self):

        '''
        iterateAllFields - Iterates over the list of fields and executes
                           iterateField for each individual field.
                         - Includes multiprocessing to improve efficiency.

        Parameters
        ----------
            None

        Inherited
        ---------
            self.pointings
            self.stars
            self.photo_path
            self.photo_tag
            self.photo_coords
            self.cm_limits

        Returns
        -------
            obsSelectionFunction - dict
                Dictionary of selection function classes for observed coordinates

        '''
        multiCore = False
        if multiCore:
            # Create processor pools for multiprocessing
            nCores = multiprocessing.cpu_count() - 1
            nCores = 2
            pool = multiprocessing.Pool( nCores )

            # List of fields in pointings database
            field_list = self.pointings.fieldID.values.tolist()

            # Locations for storage of solutions
            obsSF_dicts = {}

            # Build results into a full list of multiprocessing instances
            print("multiprocessing process for observable fields...\n")
            results = []

            """
            # Field numbering to show progress
            fieldN = 0
            fieldL = len(field_list)
            for field in field_list:

                sys.stdout.write("\rCurrent field in col-mag calculation: %s, %d/%d" % (str(field), fieldN, fieldL))
                sys.stdout.flush()

                results.append(pool.apply_async(iterateField, 
                                                args=(self.stars, self.photo_path, field,
                                                    self.photo_tag, self.photo_coords, self.pointings.loc[field],
                                                    self.cm_limits)))
                fieldN+=1

            for r in results:
                obsSF_field, field = r.get()
                obsSelectionFunction[field] = obsSF_field

            # Exit the pools as they won't be used again
            pool.close()
            pool.join()
            """

            pool = multiprocessing.Pool(processes=2)
            results = pool.map(obsMultiFunction(self.stars, self.photo_path, self.photo_tag,
                                                self.photo_coords, self.pointings, self.cm_limits), field_list)

            # Exit the pools as they won't be used again
            pool.close()
            pool.join()

            for r in results:
                obsSF_field, field = r
                obsSF_dicts[field] = vars(obsSF_field)

        else:
            # List of fields in pointings database
            field_list = self.pointings.fieldID.values.tolist()

            # Locations for storage of solutions
            obsSF_dicts = {}

            # Field numbering to show progress
            fieldN = 0
            fieldL = len(field_list)
            for field in field_list:

                sys.stdout.write("\rCurrent field in col-mag calculation: %s, %d/%d" % (str(field), fieldN, fieldL))
                sys.stdout.flush()

                obsSF_field, field = iterateField(self.stars, self.photo_path, field,
                                                self.photo_tag, self.photo_coords, self.pointings.loc[field],
                                                self.cm_limits)
                obsSF_dicts[field] = vars(obsSF_field)

                fieldN+=1

        print("Parallel done")

        return obsSF_dicts


    def createDistMhAgeInterp(self, agerng=(0,13), mhrng=(-2.5,0.5)):

        '''
        createDistMhAgeInterp - Creates a selection function in terms of age, mh, s
                              - Integrates interpolants over isochrones.

        Inherited
        ---------
            self.photo_interp: dict
                    - Dictionary of photometric interpolants in col-mag space with col-mag ranges and grid areas given

            sfFieldColMag - Converts spectroscopic interpolant and photomectric interpolant into a selection grid
                            in colour and magnitude space

            self.iso_pickle - str
                    Path to isochrone files

            self.isocolmag_pickle - str
                    Path to colour-magnitude interpolant files

            self.isointerp_exists - boolean
                    Has an interpolant over the isochrones already been created?

        **kwargs
        --------
            agemin/agemax: float (0, 13)
                    - Age range over which we want to build the selection function

            mhmin/mhmax: float (-2.5, 0.5)
                    - mh range over which we want to build the selection function

        Returns
        -------
            intrinsicSF: selfun/SFInstanceClasses class
                    - Class from which the selection function may be calculated.
                    - Calculation on age, mh, mass, s

            intrinsicIMSF: selfun/SFInstanceClasses class
                    - Class from which the selection function may be calculated
                    - Calculation on age, mh, mass

            (agemin,agemax): tuple of floats
                    - age range of the selection function

            (mhmin,mhmax): tuple of floats
                    - mh range of the selection function
        '''

        # Initialise the
        IsoCalculator = IsochroneScaling.IntrinsicToObservable()

        if not self.isointerp_exists:

            IsoCalculator.iso_pickle = self.iso_pickle
            IsoCalculator.agerng = agerng
            IsoCalculator.mhrng = mhrng

            print("Calculating isochrone distributions...")
            IsoCalculator.CreateFromIsochrones()
            print("...done\n")

            print("Pickling isochrone distributions...")
            IsoCalculator.pickleColMag(self.isocolmag_pickle)
            print("...done\n")

        else:

            print("Importing colour-magnitude isochrone interpolants...")
            IsoCalculator.LoadColMag(self.isocolmag_pickle)
            print("...done\n")

        agerng = IsoCalculator.agerng
        mhrng = IsoCalculator.mhrng
        magrng = IsoCalculator.magrng
        colrng = IsoCalculator.colrng

        # Instance of class for creating mass dependent selection functions
        intrinsicSF = SFInstanceClasses.intMassSF_isocalc()
        SFInstanceClasses.setattrs(intrinsicSF, IsoCalculator=IsoCalculator,
                                                agerng=agerng, mhrng=mhrng)
        # Instance of class for creating mass independent selection functions
        intrinsicIMFSF = SFInstanceClasses.intSF_isocalc()
        SFInstanceClasses.setattrs(intrinsicIMFSF, IsoCalculator = IsoCalculator,
                                                agerng=agerng, mhrng=mhrng)

        return intrinsicSF, intrinsicIMFSF, agerng, mhrng, magrng, colrng

    
            
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
        print("\nNote: this is iterating through 10k stars at a time.\n\
        If lots of memory available, increase N for greater efficiency.")
        df = AnglePointsToPointingsMatrix(stars, self.pointings,
                                          Phi, Th, 'SolidAngle', IDtype = self.fieldlabel_type,
                                          Nsample = 10000)
        
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

class obsMultiFunction():

    def __init__(self, stars, photo_path, photo_tag, photo_coords, 
                    pointings, cm_limits):

        self.stars = stars
        self.photo_path = photo_path
        self.photo_tag = photo_tag
        self.photo_coords = photo_coords
        self.pointings = pointings
        self.cm_limits = cm_limits

    def __call__(self, field):

        return iterateField(self.stars, self.photo_path, field, self.photo_tag, 
                            self.photo_coords, self.pointings.loc[field], self.cm_limits)
    
def iterateField(stars, photo_path, field, photo_tag, photo_coords, fieldpointing, cm_limits):

    '''
    iterateField- Generates a selection function instance for a given field

    Parameters
    ----------
        stars:

        photo_path: string
                - Location of folder of photometric catalogue files

        field - field type (str, int, float, np.float64...)
            The field in question

        photo_tag: str
            The file type of the photo files (.csv...)

        photo_coords: list of str
            Column headers of relevant columns in dataframe

        cm_limits

    Returns
    -------
        instanceSF - SFInstanceClasses class
            Class for calculating the selection function value given colour and magnitude
            for a field

        field - field type (str, int, float, np.float64...)
            The field in question

    '''
    
    database_coords = ['RA','DEC','J','K','H']
    df_coords = ['RA','Dec','appA','appB','appC']

    photofile = os.path.join(photo_path, str(field)+photo_tag)
    # Import photometric data and rename magnitude columns
    try: photo_points = pd.read_csv(photofile, compression='gzip')
    # Depends whether the stored files are gzip or not
    except IOError: photo_points = pd.read_csv(photofile)

    coords = ['appA', 'appB', 'appC']
    photo_points=photo_points.rename(index=str, 
                                     columns=dict(zip(photo_coords[2:5], coords)))
    photo_points['Colour'] = photo_points.appA - photo_points.appB

    # Select preferred survey stars
    spectro_points = stars[stars.fieldID == field]

    # Only create an interpolant if there are any points in the region
    if len(spectro_points)>0:

        """
        if len(spectro_points)>1:

            # Double range of colours and magnitudes to allow smoothing to be effective
            extra_mag = (np.max(spectro_points.appC)-np.min(spectro_points.appC))/2
            extra_col = (np.max(spectro_points.Colour)-np.min(spectro_points.Colour))/2

            # Range of colours and magnitudes used
            mag_min, mag_max = (np.min(spectro_points.appC) - extra_mag, np.max(spectro_points.appC) + extra_mag)
            col_min, col_max = (np.min(spectro_points.Colour) - extra_col, np.max(spectro_points.Colour) + extra_col)

        else:
            # If only 1 point, the max will be equivalent to the value
            mag_exact = np.max(spectro_points.appC)
            col_exact = np.max(spectro_points.Colour)

            # Range of colours and magnitudes used
            mag_min, mag_max = (mag_exact - 1, mag_exact + 1)
            col_min, col_max = (col_exact - 1, col_exact + 1)          
        """

        # Use given limits to determine boundaries of dataset
        # apparent mag upper bound
        if fieldpointing.mlBound == "NoLimit":
            mag_min = np.min(spectro_points.appC) - 2
            max_min = cm_limits[0]
        else: mag_min = fieldpointing.mlBound
        # apparent mag lower bound
        if fieldpointing.muBound == "NoLimit":
            mag_max = np.max(spectro_points.appC) + 2
            mag_max = cm_limits[1]
        else: mag_max = fieldpointing.muBound
        # colour uppper bound
        if fieldpointing.clBound == "NoLimit":
            col_min = np.min(spectro_points.Colour) - 0.1
            col_min = cm_limits[2]
        else: col_min = fieldpointing.clBound
        # colour lower bound
        if fieldpointing.cuBound == "NoLimit":
            col_max = np.max(spectro_points.Colour) + 0.1
            col_max = cm_limits[3]
        else: col_max = fieldpointing.cuBound

        # Chose only photometric survey points within the colour-magnitude region.
        photo_points = photo_points[(photo_points.appC >= mag_min)&\
                                    (photo_points.appC <= mag_max)&\
                                    (photo_points.Colour >= col_min)&\
                                    (photo_points.Colour <= col_max)]
        # If spectro points haven't been chosen from the full region, limit to this subset.
        spectro_points = spectro_points[(spectro_points.appC >= mag_min)&\
                                    (spectro_points.appC <= mag_max)&\
                                    (spectro_points.Colour >= col_min)&\
                                    (spectro_points.Colour <= col_max)]

        # Interpolate for photo data - Calculates the distribution function
        DF_interpolant, DF_gridarea, DF_magrange, DF_colrange = CreateInterpolant(photo_points,
                                                                                 (mag_min, mag_max), (col_min, col_max),
                                                                                 range_limited=True,
                                                                                 datatype = "photo")
        # Interpolate for spectro data - Calculates the selection function
        SF_interpolant, SF_gridarea, SF_magrange, SF_colrange = CreateInterpolant(spectro_points,
                                                                                  (mag_min, mag_max), (col_min, col_max),
                                                                                  datatype = "spectro", photoDF=DF_interpolant)

        # Store information inside an SFInstanceClasses.observableSF instance where the selection function is calculated.
        instanceSF = SFInstanceClasses.observableSF(field)
        SFInstanceClasses.setattrs(instanceSF,
                                    DF_interp = DF_interpolant,
                                    DF_gridarea = DF_gridarea,
                                    DF_magrange = DF_magrange,
                                    DF_colrange = DF_colrange,
                                    SF_interp = SF_interpolant,
                                    SF_gridarea = SF_gridarea,
                                    SF_magrange = SF_magrange,
                                    SF_colrange = SF_colrange,
                                    grid_points = True)


    else:
        # There are no stars on the field plate
        instanceSF = SFInstanceClasses.observableSF(field)
        instanceSF.grid_points = False
        

    return instanceSF, field

def CreateInterpolant(points,
                      mag_range, col_range,
                      Grid = False, range_limited=False,
                      datatype="", photoDF=None):

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
                - Interpolant of the spectroscopic survey stars on the field in col-mag space

        
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

    # What fitting process do you want to use for the data
    Process = "Poisson"

    # Optimum Poisson likelihood process
    if Process == "Poisson":

        if datatype == "photo":
            profile = PoissonLikelihood(points, mag_range, col_range,
                                        'appC', 'Colour',
                                        datatype=datatype)
        elif datatype == "spectro":
            profile = PoissonLikelihood(points, mag_range, col_range,
                                        'appC', 'Colour',
                                        datatype=datatype, photoDF=photoDF)            


        grid_area = np.nan
        return profile, grid_area, mag_range, col_range

    # Ratio of number of stars in the field.
    elif Process == "Number":

        grid_area = np.nan
        profile = FlatRegion(len(points), mag_range, col_range)

        return profile, grid_area, mag_range, col_range

    # Histogram grid density process
    elif Process == "Grid":
        Nside_grid = 8
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

# Used when Process == Poisson
def PoissonLikelihood(points,
                     mag_range, col_range,
                     mag_label, col_label,
                     datatype="", photoDF=None):
    '''
    PoissonLikelihood

    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    # Make copy of points in order to save gc.collect runtime
    points = pd.DataFrame(np.copy(points), columns=list(points))

    x = getattr(points, mag_label)
    y = getattr(points, col_label)

    modelType = 'GMM'

    if modelType == 'GMM':

        # Number of Gaussian components
        if datatype == "spectro": nComponents = 1
        elif datatype == "photo": nComponents = 2

        # Generate the model
        model = StatisticalModels.GaussianMM(x, y, nComponents, mag_range, col_range)
        # Add in SFxDF<DF constraint for the spectrograph distribution
        if datatype == "spectro": 
            model.photoDF, model.priorDFbool = (photoDF, True)
        model.runningL = False
        model.optimizeParams()
        # Test integral if you want to see the value/error in the integral when calculated
        # model.testIntegral()

    elif modelType == 'Grid':

        # Number of grid cells
        nx, ny = 8, 9
        model = StatisticalModels.PenalisedGridModel(x, y, (nx,ny), mag_range, col_range)
        model.optimizeParams()
        model.generateInterpolant()

        print('...complete')
    else:
        raise ValueError("A valid modelType has not been provided - either \"Gaussian\" for GMM or \"Grid\" for PenalisedGridModel")

    return model

# Used when Process == "Number"
class FlatRegion:

    def __init__(self, value, rangex, rangey):
        self.value = value
        self.rangex = rangex
        self.rangey = rangey

    def __call__(self, (x, y)):

        result = np.zeros(np.shape(x))
        result[(x>self.rangex[0]) & \
                (x<self.rangex[1]) & \
                (y>self.rangey[0]) & \
                (y<self.rangey[1])] = self.value

        return result

# Used when Process == Grid
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

def fieldInterp(fieldInfo, agegrid, mhgrid, sgrid, 
                age_mh, col, mag, weight, obsSF,
                mass_sf=False, massgrid=None,
                fieldN=0, fieldL=0):
                #spectro, photo):

    '''
    fieldInterp - Converts grids of colour and magnitude coordinates and a 
                  colour-magnitude interpolant into the age, mh, s selection
                  function interpolant.

    Parameters
    ----------
        fieldInfo: single row Dataframe
                - fieldID and photo_bool information on the given field

        agegrid, mhgrid, sgrid: 3D array
                - Distribution of coordinates over which the selection function
                  will be generated

        obsSF: function instance from FieldInterpolator class
                sfFieldColMag - Converts survey interpolant and 2MASS interpolant into a selection grid
                                in colour and magnitude space

        age_mh: Dataframe
                - Contains all unique age-metalicity values as individual rows which are then unstacked to
                  a matrix to allow for efficient calculation of the selection function.

        col: array
                - Matrix of colour values over the colour-magnitude space

        mag: array
                - Matrix of H magnitude values over the colour-magnitude space

        weight: array
                - Matrix of weightings of isochrone points so that the IMF is integrated over

    Dependencies
    ------------
        RegularGridInterpolator

    Returns
    -------
    if photo_bool:
        sfinterpolant

    else:
        RGI([], 0)
    '''

    # Import field data
    fieldID = fieldInfo['fieldID']


    # True if there were any stars in the field pointing
    if obsSF.grid_points:
        # For fields with RAVE-TGAS stars
        sys.stdout.write("\rCurrent field being interpolated: %s, %d/%d" % (fieldID, fieldL, fieldN))
        sys.stdout.flush()

        sfprob = np.zeros_like(col)

        # Make sure values fall within interpolant range for colour and magnitude
        # Any points outside the range will provide a 0 contribution
        col[np.isnan(col)] = np.inf
        mag[np.isnan(mag)] = np.inf
        bools = (col>obsSF.DF_colrange[0])&\
                (col<obsSF.DF_colrange[1])&\
                (mag>obsSF.DF_magrange[0])&\
                (mag<obsSF.DF_magrange[1])
        # np.nan values provide a nan contribution
        sfprob[bools] = obsSF((mag[bools],col[bools]))

        # Transform to selection grid (age,mh,s) and interpolate to get plate selection function
        age_mh['nonintegrand'] = sfprob.tolist()

        sfgrid = np.array(age_mh[['nonintegrand']].unstack()).tolist()
        sfgrid = np.array(sfgrid)

        # Expand grids to account for central coordinates
        sfgrid, age4grid = extendGrid(sfgrid, agegrid, axis=0, x_lbound=True, x_lb=0.)
        sfgrid, mh4grid = extendGrid(sfgrid, mhgrid, axis=1)
        sfgrid, mass4grid = extendGrid(sfgrid, massgrid, axis=2, x_lbound=True, x_lb=0.)
        sfgrid, s4grid = extendGrid(sfgrid, sgrid, axis=3, x_lbound=True, x_lb=0.)

        sf4interpolant = RGI((age4grid,mh4grid,mass4grid,s4grid),sfgrid, bounds_error=False, fill_value=0.0)
        del(age4grid,mh4grid,mass4grid,s4grid,sfgrid)
        gc.collect()

        # Include weight from IMF and density of mass points per isochrone
        sfprob *= weight
        # Integrating (sum) selection probabilities to get in terms of age,mh,s
        integrand = np.sum(sfprob, axis=1)

        # Transform to selection grid (age,mh,s) and interpolate to get plate selection function
        age_mh['integrand'] = integrand.tolist()

        sfgrid = np.array(age_mh[['integrand']].unstack()).tolist()
        sfgrid = np.array(sfgrid)

        # Expand grids to account for central coordinates
        sfgrid, age3grid = extendGrid(sfgrid, agegrid, axis=0, x_lbound=True, x_lb=0.)
        sfgrid, mh3grid = extendGrid(sfgrid, mhgrid, axis=1)
        sfgrid, s3grid = extendGrid(sfgrid, sgrid, axis=2, x_lbound=True, x_lb=0.)

        sf3interpolant = RGI((age3grid,mh3grid,s3grid),sfgrid, bounds_error=False, fill_value=0.0)


    else:
        # For fields with no RAVE-TGAS stars - 0 value everywhere in field
        print('No stars in field: '+str(fieldID))
        sf3interpolant, sf4interpolant = (RGI([], 0), RGI([], 0))

    return sf3interpolant, sf4interpolant, fieldID


def sfFieldColMag(field, col, mag, weight, spectro, photo):

    '''
    sfFieldColMag - Converts spectro interpolant and 2MASS interpolant into a selection grid
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
        self.spectro_interp: dict
                - Dictionary of spectroscopic spectro interpolants in col-mag space with col-mag ranges and grid areas given

        self.tmass_interp: dict
                - Dictionary of photometric spectro interpolants in col-mag space with col-mag ranges and grid areas given

    Returns
    -------
        grid: array
                - Array with same dimensions as col/mag giving the ratio of spectro/2MASS interpolants normalised
                  by the grid areas.
    '''
    
    grid = np.zeros_like(col)
    
    # If dictionary contains nan values, the field contains no stars
    if ~np.isnan(spectro['grid_area']):

        bools = (col>spectro['col_range'][0])&(col<spectro['col_range'][1])&\
                (mag>spectro['mag_range'][0])&(mag<spectro['mag_range'][1])

        grid[bools] = (spectro['interp']((mag[bools],col[bools]))/spectro['grid_area'])/\
                      (photo['interp']((mag[bools],col[bools]))/photo['grid_area'])   
        
        grid[grid==np.inf]=0.
        grid[np.isnan(grid)] = 0.

    # Weight the grid to correspond to IMF weightings for each star
    prob = grid*weight

    return prob

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



