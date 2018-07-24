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
import healpy as hp
from itertools import product
import re, dill, pickle, multiprocessing
import cProfile, pstats, time
import sys, os
from scipy.interpolate import RegularGridInterpolator as RGI

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

from seestar import ArrayMechanics as AM
from seestar import StatisticalModels
from seestar import FieldUnions
from seestar import SFInstanceClasses
from seestar import IsochroneScaling
from seestar import surveyInfoPickler
from seestar import AngleDisks
from seestar import FieldAssignment

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
    
    def __init__(self, pickleFile):

        # Check what components of the selection function already exist
        gen_intsf, use_intsf, gen_obssf, use_obssf, use_isointerp = path_check(pickleFile)

        # Load survey information from pickleFile
        fileinfo = surveyInfoPickler.surveyInformation(pickleFile)
        self.fileinfo = fileinfo

        spectro_coords = fileinfo.spectro_coords
        spectro_path = fileinfo.spectro_path

        self.photo_coords = fileinfo.photo_coords
        self.photo_path = fileinfo.photo_path

        self.field_coords = fileinfo.field_coords
        field_path = fileinfo.field_path

        obsSF_pickle_path = fileinfo.obsSF_pickle_path
        sf_pickle_path = fileinfo.sf_pickle_path

        self.photo_tag = fileinfo.photo_tag
        self.fieldlabel_type = fileinfo.field_dtypes[0]

        self.iso_pickle = fileinfo.iso_pickle_path
        self.isocolmag_pickle = fileinfo.iso_interp_path
        self.use_isointerp = use_isointerp

        self.photo_model = fileinfo.photo_model
        self.spectro_model = fileinfo.spectro_model

        # Import the dataframe of pointings        
        pointings = self.ImportDataframe(field_path, self.field_coords, 
                                         data_source='field')
        pointings = pointings.set_index(self.field_coords[0], drop=False)
        pointings = pointings.drop_duplicates(subset = self.field_coords[0])
        #pointings = pointings[pointings.fieldID==2001.0]
        self.pointings = pointings

        if gen_intsf: # If we want to generate an intrinsic selection function    
            if use_intsf: # Use a premade intrinsic selection function
                # Unpickle survey selection function
                print("Unpickling survey selection function...")
                with open(sf_pickle_path, "rb") as f:
                    instanceSF_dict, instanceIMFSF_dict, self.agerng, self.mhrng, \
                    magrng, colrng = pickle.load(f)
                print("...done.\n") 

                # Load class instance from saved dictionary
                self.instanceSF = SFInstanceClasses.intrinsicSF()
                self.instanceIMFSF = SFInstanceClasses.intrinsicIMFSF()
                SFInstanceClasses.setattrs(self.instanceSF, **instanceSF_dict)
                SFInstanceClasses.setattrs(self.instanceIMFSF, **instanceIMFSF_dict)

                # Can't set cm_limits because they use absolute magnitude
                self.cm_limits = None
                #self.cm_limits = (magrng[0], magrng[1], colrng[0], colrng[1])

            else:

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
                
                # Can't set cm_limits because they use absolute magnitude
                self.cm_limits = None
                #self.cm_limits = (magrng[0], magrng[1], colrng[0], colrng[1])
                
                print("...done.\n")
        else: # Still need cm_limits for obsSF
            self.cm_limits = None

        if gen_obssf: # We want an observable selection function
            if use_obssf: # Use the premade one
                # Once Colour Magnitude selection functions have been created
                # Unpickle colour-magnitude interpolants
                print("Unpickling colour-magnitude interpolant dictionaries...")
                with open(obsSF_pickle_path, "rb") as f:
                    obsSF_dicts = pickle.load(f)
                print("...done.\n")
            else: # Create new observable selection function
                print('Importing data for Colour-Magnitude Field interpolants...')
                self.spectro_df = self.ImportDataframe(spectro_path, spectro_coords)
                print("...done.\n")
            
                print('Creating Colour-Magnitude Field interpolants...')
                obsSF_dicts = self.iterateAllFields()
                print('\nnow pickling them...')
                with open(obsSF_pickle_path, 'wb') as handle:
                    pickle.dump(obsSF_dicts, handle)
                print("...done\n.")
            # Initialise dictionary
            self.obsSF = SFInstanceClasses.obsSF_dicttoclass(obsSF_dicts)
        
    def __call__(self, catalogue, method='intrinsic', 
                coords = ['age', 'mh', 's', 'mass'], 
                angle_coords=['phi','theta'],
                rng_phi=(0, 2*np.pi), rng_th=(-np.pi/2, np.pi/2)):

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
        elif method=='intrinsic':
            SFcalc = lambda field, df: self.instanceSF( (df[coords[0]], df[coords[1]], df[coords[3]], df[coords[2]]), self.obsSF[field] )
        else: raise ValueError('Method is unknown')

        # Labels for angles in pointings (reset in ImportData)
        point_coords = ['phi','theta']

        if self.fileinfo.style == 'mf':
            # Drop column which will be readded if this has been calculated before.
            if 'SFprob' in list(catalogue): catalogue = catalogue.drop('SFprob', axis=1)
            print('Calculating all SF values...')
            catalogue = FieldUnions.GenerateMatrices(catalogue, self.pointings, 
                                                    angle_coords, point_coords, 'halfangle', 
                                                    SFcalc, IDtype=self.fieldlabel_type)
            print('...done')
            
            # The SF probabilities are used to calculate the field union.
            print('Calculating union contribution...')
            FUInstance = FieldUnions.FieldUnion()
            catalogue['union'] = FUInstance(catalogue.SFprob)
            print('...done')
        elif self.fileinfo.style == 'as':
            npixel = len(self.pointings)
            nside = hp.npix2nside( npixel )
            # Assign stars to fields then calculate selection functions
            theta = catalogue[angle_coords[1]]
            phi = catalogue[angle_coords[2]]
            catalogue[self.field_coords[0]] = FieldAssignment.labelStars(theta, phi, rng_th, rng_phi, nside)

            # Calculate selection function
            catalogue['sfprob'] = 0.
            for field in self.pointings[self.field_coords[0]]:
                # Calculate probabilities for stars on the field
                isfield = catalogue[self.field_coords[0]] == field
                catalogue['sfprob'][isfield] = SFcalc(field, catalogue[isfield])

        return catalogue
        
        
    def ImportDataframe(self, path,
                        coord_labels, 
                        data_source = 'spectro'):

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
            data_source: string ('spectro')
                    - 'spectro', assumes we are importing a database of stars
                        (so it imports magnitudes aswell)
                    - 'field': assumes we are importing a database of fields
                        (so it imports halfangle and calculates halfangle too)

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
        df = getattr(pd, 'read_'+filetype)(path, 
                                             usecols = coord_labels)

        if data_source=='spectro': 
            coords = ['fieldID','phi', 'theta', 'appA', 'appB', 'appC']
        elif data_source=='field':
            coords = ['fieldID', 'phi', 'theta', 'halfangle', 'Magmin', 'Magmax', 'Colmin', 'Colmax']
            #coords.extend(['halfangle'])

        # Replace given coordinates with standardised ones
        print(dict(zip(coord_labels, coords)))
        df = df.rename(index=str, columns=dict(zip(coord_labels, coords)))

        # Remove any null values from df
        if data_source=='spectro':
            full_length = len(df)
            for coord in coords: df = df[pd.notnull(df[coord])]
            used_length = len(df)
            print("Filtering for null values in %s: Total star count = %d. Filtered star count = %d. %d stars removed with null values" % \
                    (data_source, full_length, used_length, full_length-used_length))

        # Correct units
        """if (angle_units == 'degrees') & \
           (coordinates == 'Equatorial'): df.RA, df.Dec = df.RA*np.pi/180, df.Dec*np.pi/180
        elif (angle_units == 'degrees') & \
             (coordinates == 'Galactic'): df.l, df.b = df.l*np.pi/180, df.b*np.pi/180
        elif angle_units in ('rad','radians'): pass
        else: raise ValueError ("I don't understand the units specified")

        # Include Galactic Coordinates
        if coordinates == 'Equatorial': df['l'], df['b'] = AngleDisks.EquatToGal(df.RA, df.Dec)
        if coordinates == 'Galactic': df['RA'], df['Dec'] = AngleDisks.GalToEquat(df.l, df.b)"""

        if data_source=='spectro': 
            df['Colour'] = df.appA - df.appB

            # Magnitude and colour ranges from full sample
            mag_range = (np.min(df.appC), np.max(df.appC))
            col_range = (np.min(df.Colour), np.max(df.Colour))

            #Save star results in the class variables
            self.mag_range = mag_range
            self.col_range = col_range

            print(mag_range, col_range)

            return df

        elif data_source == 'field':

            #Save field df in the class variable
            return df
            
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
            self.spectro_df
            self.photo_path
            self.photo_tag
            self.photo_coords
            self.cm_limits

        Returns
        -------
            obsSelectionFunction - dict
                Dictionary of selection function classes for observed coordinates

        '''
        start = time.time()
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
                                                args=(self.spectro_df, self.photo_path, field,
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
            results = pool.map(obsMultiFunction(self.spectro_df, self.photo_path, self.photo_tag,
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
            tnow = 0
            tleft = 0
            for field in field_list:

                fieldN+=1
                sys.stdout.write("\rCurrent field in col-mag calculation: %s, %d/%d, Time: %dm, Left: %dm" % (str(field), fieldN, fieldL, int(tnow), int(tleft)))
                sys.stdout.flush()

                obsSF_field, field = iterateField(self.spectro_df, self.photo_path, field,
                                                self.photo_tag, self.photo_coords, self.pointings.loc[field],
                                                cm_limits=self.cm_limits, spectro_model=self.spectro_model, photo_model=self.photo_model)
                obsSF_dicts[field] = vars(obsSF_field)

                tnow = (time.time() - start)/60.
                tleft = tnow*(float(fieldL)/fieldN - 1)

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

            self.use_isointerp - boolean
                    Shall we use a premated interpolant of the isochrones?

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

        if self.use_isointerp:
            print("Importing colour-magnitude isochrone interpolants...")
            IsoCalculator.LoadColMag(self.isocolmag_pickle)
            print("...done")
        else:
            IsoCalculator.iso_pickle = self.iso_pickle
            IsoCalculator.agerng = agerng
            IsoCalculator.mhrng = mhrng

            print("Calculating isochrone distributions...")
            IsoCalculator.CreateFromIsochrones()
            print("...done")

            print("Pickling isochrone distributions...")
            IsoCalculator.pickleColMag(self.isocolmag_pickle)
            print("...done")

        agerng = IsoCalculator.agerng
        mhrng = IsoCalculator.mhrng
        # This is the ABSOLUTE magnitude range
        magrng = IsoCalculator.magrng
        colrng = IsoCalculator.colrng

        # Instance of class for creating mass dependent selection functions
        intrinsicSF = SFInstanceClasses.intrinsicSF()
        SFInstanceClasses.setattrs(intrinsicSF, IsoCalculator=IsoCalculator)
        # Instance of class for creating mass independent selection functions
        intrinsicIMFSF = SFInstanceClasses.intrinsicIMFSF()
        SFInstanceClasses.setattrs(intrinsicIMFSF, IsoCalculator = IsoCalculator)

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
            self.spectro_df: Dataframe
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

        points = self.spectro_df[self.spectro_df.field_ID == field]
        
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
    
    def PointsToPointings(self, stars, Phi='phi', Th='theta'):

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

        # Decide based on available memory how many stars to iterate at once
        Nsample = FieldAssignment.iterLimit(len(pointings))
        df = AM.AnglePointsToPointingsMatrix(stars, self.pointings,
                                          Phi, Th, 'halfangle', IDtype = self.fieldlabel_type,
                                          Nsample=Nsample)
        
        return df

    def PlotPlates(self, **kwargs):

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
            PhiRad = np.copy(self.pointings.phi)
            ThRad = np.copy(self.pointings.theta)      
        else:
            PhiRad = np.copy(self.pointings.loc[fieldIDs].phi)
            ThRad = np.copy(self.pointings.loc[fieldIDs].theta)

        halfangle = np.copy(self.pointings.halfangle)
        SA = 2*np.pi*(1 - np.cos(halfangle))

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='mollweide')

        for i in range(len(PhiRad)):
            Phi, Th = GenerateCircle(PhiRad[i],ThRad[i],SA[i])
            PlotDisk(Phi, Th, ax)

class obsMultiFunction():

    def __init__(self, spectro_df, photo_path, photo_tag, photo_coords, 
                    pointings, cm_limits):

        self.spectro_df = spectro_df
        self.photo_path = photo_path
        self.photo_tag = photo_tag
        self.photo_coords = photo_coords
        self.pointings = pointings
        self.cm_limits = cm_limits

    def __call__(self, field):

        return iterateField(self.spectro_df, self.photo_path, field, self.photo_tag, 
                            self.photo_coords, self.pointings.loc[field], self.cm_limits)
    
def iterateField(spectro, photo_path, field, photo_tag, photo_coords, fieldpointing, cm_limits=None, 
                    spectro_model=('GMM', 1), photo_model=('GMM', 2)):

    '''
    iterateField- Generates a selection function instance for a given field

    Parameters
    ----------
        spectro:

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
    spectro_points = spectro[spectro.fieldID == field]

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
        if fieldpointing.Magmin == "NoLimit":
            if cm_limits is None: mag_min = np.min(spectro_points.appC) - 1
            else: mag_min = cm_limits[0]
        else: mag_min = fieldpointing.Magmin
        # apparent mag lower bound
        if fieldpointing.Magmax == "NoLimit":
            if cm_limits is None: mag_max = np.max(spectro_points.appC) + 1
            else: mag_max = cm_limits[1]
        else: mag_max = fieldpointing.Magmax
        # colour uppper bound
        if fieldpointing.Colmin == "NoLimit":
            if cm_limits is None: col_min = np.min(spectro_points.Colour) - 0.1
            else: col_min = cm_limits[2]
        else: col_min = fieldpointing.Colmin
        # colour lower bound
        if fieldpointing.Colmax == "NoLimit":
            if cm_limits is None: col_max = np.max(spectro_points.Colour) + 0.1
            else: col_max = cm_limits[3]
        else: col_max = fieldpointing.Colmax

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
        DF_model, DF_magrange, DF_colrange = CreateInterpolant(photo_points,
                                                                     (mag_min, mag_max), (col_min, col_max),
                                                                     range_limited=True,
                                                                     datatype = "photo", modelinfo=photo_model)
        # Interpolate for spectro data - Calculates the selection function
        SF_model, SF_magrange, SF_colrange = CreateInterpolant(spectro_points,
                                                              (mag_min, mag_max), (col_min, col_max),
                                                              datatype = "spectro", photoDF=DF_model, modelinfo=spectro_model)

        # Store information inside an SFInstanceClasses.observableSF instance where the selection function is calculated.
        instanceSF = SFInstanceClasses.observableSF(field)
        SFInstanceClasses.setattrs(instanceSF,
                                    DF_model = vars(DF_model),
                                    DF_magrange = DF_magrange,
                                    DF_colrange = DF_colrange,
                                    SF_model = vars(SF_model),
                                    SF_magrange = SF_magrange,
                                    SF_colrange = SF_colrange)

    else:
        # There are no stars on the field plate
        instanceSF = SFInstanceClasses.observableSF(field)

        # Create class instances so that the models will still work.
        DF_model = StatisticalModels.Empty()
        SF_model = StatisticalModels.Empty()

        SFInstanceClasses.setattrs(instanceSF,
                                    DF_model = vars(DF_model),
                                    SF_model = vars(SF_model),
                                    DF_magrange = (0,0),
                                    DF_colrange = (0,0),
                                    SF_magrange = (0,0),
                                    SF_colrange = (0,0))
        

    return instanceSF, field

def CreateInterpolant(points,
                      mag_range, col_range,
                      Grid = False, range_limited=False,
                      datatype="", photoDF=None, modelinfo=('GMM', 1)):

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
        model: StatisticalModels: GaussianMM, PenalisedGrid or FlatRegion instance
                - Interpolant of the spectroscopic survey stars on the field in col-mag space
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
            model = PoissonLikelihood(points, mag_range, col_range,
                                        'appC', 'Colour',
                                        datatype=datatype, modelinfo=modelinfo)
        elif datatype == "spectro":
            model = PoissonLikelihood(points, mag_range, col_range,
                                        'appC', 'Colour',
                                        datatype=datatype, photoDF=photoDF, modelinfo=modelinfo)

        return model, mag_range, col_range

    # Ratio of number of stars in the field.
    elif Process == "Number":

        model = StatisticalModels.FlatRegion(len(points), mag_range, col_range)

        return model, mag_range, col_range

# Used when Process == Poisson
def PoissonLikelihood(points,
                     mag_range, col_range,
                     mag_label, col_label,
                     datatype="", photoDF=None, modelinfo=('GMM', 1)):
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

    modelType = modelinfo[0]

    if modelType == 'GMM':

        # Number of Gaussian components
        nComponents = modelinfo[1]

        # Generate the model
        model = StatisticalModels.GaussianEM(x=x, y=y, nComponents=nComponents, rngx=mag_range, rngy=col_range)
        #model = StatisticalModels.GaussianMM(x, y, nComponents, mag_range, col_range)
        # Add in SFxDF<DF constraint for the spectrograph distribution
        if datatype == "spectro": 
            model.photoDF, model.priorDF = (photoDF, True)
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
        sfgrid, age4grid = AM.extendGrid(sfgrid, agegrid, axis=0, x_lbound=True, x_lb=0.)
        sfgrid, mh4grid = AM.extendGrid(sfgrid, mhgrid, axis=1)
        sfgrid, mass4grid = AM.extendGrid(sfgrid, massgrid, axis=2, x_lbound=True, x_lb=0.)
        sfgrid, s4grid = AM.extendGrid(sfgrid, sgrid, axis=3, x_lbound=True, x_lb=0.)

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
        sfgrid, age3grid = AM.extendGrid(sfgrid, agegrid, axis=0, x_lbound=True, x_lb=0.)
        sfgrid, mh3grid = AM.extendGrid(sfgrid, mhgrid, axis=1)
        sfgrid, s3grid = AM.extendGrid(sfgrid, sgrid, axis=2, x_lbound=True, x_lb=0.)

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
        displacement = AM.AngleSeparation(points[Phistr], points[Thstr], Phi, Th)
        field = points[displacement == displacement.min()]['fieldID'].iloc[0]
        
        fieldlist.append(field)
        
    return fieldlist

def path_check(pickleFile):

    '''
    path_check - Generates a set of booleans which determine which parts of the selection function
        need to be loaded in and which files can be pulled from.

    Parameters
    ----------
        pickleFile: str
            - Path to survey information pickle file

    Returns
    -------
        gen_intsf: bool
            - Does an intrinsic selection function need to be generated?
        use_intsf: bool
            - Can we load the intrinsic SF from a premade file?
        gen_obssf: bool
            - Does an observable selection function need to be generated?
        use_obssf: bool
            - Can we load the observable SF from a premade file?
        use_isointerp: bool
            - Can we load the isochrone information from the interp pickle file?
    '''

    fileinfo = surveyInfoPickler.surveyInformation(pickleFile)

    # INTRINSIC AND OBSERVABLE SELECTION FUNCTIONS
    good_type = False
    while not good_type:
        # Intrinsic or observable
        sftype = raw_input("Would you like the selection function in: a) observable, b) intrinsic, c) both? (return a, b or c)")

        if sftype in ('b', 'c'): # Intrinsic or both
            good_type = True
            gen_intsf = True
            # Check intrinsic SF
            if os.path.exists(fileinfo.sf_pickle_path):

                good = False
                while not good:
                    use_sf = raw_input("Path to intrinsic SF (%s) exists. Load SF in from here? (y/n)" % fileinfo.sf_pickle_fname)
                    if use_sf == 'y':
                        use_intsf = True
                        good = True
                    elif use_sf == 'n':
                        print("Will generate a new intrinsic SF.")
                        use_intsf = False
                        good = True
                    else: pass # Good stays false cause input wasn't y/n
            else: use_intsf = False

            if sftype == 'b': # Intrinsic only
                if use_intsf: # If intrinsic SF already exists, don't need observable SF
                    gen_obssf = False
                    use_obssf = False # must be false if gen_obssf is false
                else: # No intrinsic SF so need to create one
                    gen_obssf = True
                    good = False
                    while not good:
                        if os.path.exists(fileinfo.obsSF_pickle_path):
                            use_sf = raw_input("Path to observable SF (%s) exists. Use this to ? (y/n)" % fileinfo.obsSF_pickle_fname)
                            if use_sf == 'y':
                                use_obssf = True
                                good = True
                            elif use_sf == 'n':
                                use_obssf = False
                                good = True
                            else: pass # Good stays false cause input wasn't y/n
                        else: # No premade observed selection function
                            print('No observed SF found at %s, therefore will start from scratch.' % fileinfo.obsSF_pickle_path)
                            use_obssf = False
                            good = True

            elif sftype == 'c': # Intrinsic and observable
                gen_obssf = True
                good = False
                while not good:
                    if os.path.exists(fileinfo.obsSF_pickle_path):
                        use_sf = raw_input("Path to observable SF (%s) exists. Use this to ? (y/n)" % fileinfo.obsSF_pickle_fname)
                        if use_sf == 'y':
                            use_obssf = True
                            good = True
                        elif use_sf == 'n':
                            use_obssf = False
                            good = True
                        else: pass # Good stays false cause input wasn't y/n
                    else: # No premade observed selection function
                        print("No observed SF found at %s, we'll build observable seleciton function from scratch." % fileinfo.obsSF_pickle_path)
                        use_obssf = False
                        good = True

        elif sftype == 'a': # Just observable
            good_type = True
            gen_obssf = True # Must create an observable selection function
            gen_intsf = False # No need for and intrinsic SF
            use_intsf = False # Must be false if gen_intsf is false
            good = False
            while not good:
                if os.path.exists(fileinfo.obsSF_pickle_path):
                    use_sf = raw_input("Path to observable SF (%s) exists. Use this to ? (y/n)" % fileinfo.obsSF_pickle_fname)
                    if use_sf == 'y':
                        use_obssf = True
                        good = True
                    elif use_sf == 'n':
                        use_obssf = False
                        good = True
                    else: pass # Good stays false cause input wasn't y/n
                else: # No premade observed selection function
                    print("No observed SF found at %s, we'll build observable seleciton function from scratch." % fileinfo.obsSF_pickle_path)
                    use_obssf = False
                    good = True

        else: pass # sf_type is not in a, b, or c

    # ISOCHRONES
    if gen_intsf & (not use_intsf): # Only need isochrones for generating an intrinsic Selection Function.
        if os.path.exists(fileinfo.iso_interp_path):
            print("Path to interpolated isochrones (%s) exists. These will be used." % fileinfo.iso_interp_file)
            use_isointerp = True
        elif os.path.exists(fileinfo.iso_data_path):
            print("No interpolated isochrones. They will be generated from %s." % fileinfo.iso_data_file)
            use_isointerp = False
        else: raise IOError('No isochrone files at %s or %s' % (fileinfo.iso_interp_path, fileinfo.iso_data_path))
    else: use_isointerp = 'na'

    # Model
    print("The spectro model description is:"+str(fileinfo.spectro_model))
    print("The photo model description is:"+str(fileinfo.photo_model))    


    print('')

    return gen_intsf, use_intsf, gen_obssf, use_obssf, use_isointerp
