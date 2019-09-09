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
'''

import numpy as np
import pandas as pd
import healpy as hp
from itertools import product
import re, pickle, multiprocessing
import cProfile, pstats, time
import sys, os
from scipy.interpolate import RegularGridInterpolator as RGI

if sys.version<'3': # If python 2, use raw_input, not input
    input = raw_input

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

#from seestar import ArrayMechanics as AM, StatisticalModels, FieldUnions, SFInstanceClasses, FieldAssignment
sys.path.append('/home/andy/Documents/Research/SF/GitRepo/seestar/')
import ArrayMechanics as AM, StatisticalModels, FieldUnions, SFInstanceClasses, FieldAssignment

class SFGenerator():

    '''
    FieldInterpolants - Class for building a dictionary of interpolants for each of the survey fields
                        The interpolants are used for creating field selection functions

    Parameters
    ----------
        pickleFile - str - path to pickled instance of class with file names and coordinate lists

    kwargs
    ------
        ncores=1: int
            - Number of available cores to use

        memory: float
            - Number of Gb of memory available across all cores

    Functions
    ---------
        StarDataframe - Creates a dataframe of relevant relevant information on stars used to create  interpolants

        PointingsDataframe - Creates a dataframe of relevant relevant information on the coordinates of RAVE Fields

        IterateFields - Runs CreateInterpolant for each Field in the database

    Returns
    -------
        self.interpolants: dictionary of floats and interpolants
                - Dictionary with an entry for each field with the interpolant, mag_range, col_range, grid_area

    Dependencies
    ------------
        EquatToGal
        AnglePointsToPointingsMatrix
    '''

    def __init__(self, get_spectro, get_photo, pointings,
                    spectro_model=('GMM',3), photo_model=('GMM',3),
                    ncores=0, memory=1, style='multiobj', **kwargs):

        # Get data
        self.get_spectro = get_spectro
        self.get_photo = get_photo

        # Spectrograph type/style (multiobj e.g. APOGEE, allsky e.g. Gaia RV)
        self.style=style

        # Model
        print("The spectro model description is:"+str(spectro_model))
        print("The photo model description is:"+str(photo_model))
        self.photo_model = photo_model
        self.spectro_model = spectro_model

        self.varrngs = {'agerng':None, 'mhrng':None, 'magrng':None, 'colrng':None}

        # Hold number of cores and memory available to use
        self.ncores=ncores
        # Memory isn't actually being used in this, might try to build it in
        self.memory=memory

        field_coords = ['fieldID','phi','theta','halfangle','Magmin', 'Magmax', 'Colmin', 'Colmax']
        #pointings = pd.DataFrame(np.array(pointings),columns=field_coords)
        pointings = pointings.set_index('fieldID', drop=False)
        pointings = pointings.drop_duplicates(subset = 'fieldID')
        self.pointings = pointings
        self.field_coords = field_coords

        #self.fieldlabel_type = pointings['fieldID'].dtypes
        self.fieldlabel_type = type(pointings.fieldID.iloc[0])

        self.cm_limits = None

        self.obsSF_dicts={}

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

        catalogue = catalogue[coords+angle_coords]

        if method=='observable':
            SFcalc = lambda field, df: np.array( self.obsSF[field]((df[coords[0]], df[coords[1]])) )
        elif method=='IMFint':
            SFcalc = lambda field, df: self.instanceIMFSF( (df[coords[0]], df[coords[1]], df[coords[2]]), self.obsSF[field] )
        elif method=='intrinsic':
            SFcalc = lambda field, df: self.instanceSF( (df[coords[0]], df[coords[1]], df[coords[3]], df[coords[2]]), self.obsSF[field] )
        else: raise ValueError('Method is unknown')

        # Labels for angles in pointings (reset in ImportData)
        point_coords = ['phi','theta']

        if self.style == 'multiobj':
            # Drop column which will be readded if this has been calculated before.
            if 'SFprob' in list(catalogue): catalogue = catalogue.drop('SFprob', axis=1)
            print('Calculating all SF values...')
            catalogue = FieldUnions.GenerateMatrices(catalogue, self.pointings,
                                                    angle_coords, point_coords, 'halfangle',
                                                    SFcalc, IDtype=self.fieldlabel_type)
            print('...done')
            #print catalogue.SFprob

            # The SF probabilities are used to calculate the field union.
            print('Calculating union contribution...')
            FUInstance = FieldUnions.FieldUnion()
            catalogue['union'] = FUInstance(catalogue.SFprob)
            print('...done')
        elif self.style == 'allsky':
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

    def load_intSF(self, intSF_path):

        # Unpickle intrinsic selection function
        print("Unpickling intrinsic selection function...")
        with open(intSF_path, "rb") as f:
            instanceSF_dict, instanceIMFSF_dict, self.agerng, self.mhrng, \
            magrng, colrng = pickle.load(f)
        print("...done.\n")
        self.varrngs.update({'agerng':self.agerng, 'mhrng':self.mhrng, 'magrng':magrng, 'colrng':colrng})

        # Load class instance from saved dictionary
        self.instanceSF = SFInstanceClasses.intrinsicSF()
        self.instanceIMFSF = SFInstanceClasses.intrinsicIMFSF()
        SFInstanceClasses.setattrs(self.instanceSF, **instanceSF_dict)
        SFInstanceClasses.setattrs(self.instanceIMFSF, **instanceIMFSF_dict)

        # Can't set cm_limits because they use absolute magnitude
        self.cm_limits = None
        #self.cm_limits = (magrng[0], magrng[1], colrng[0], colrng[1])

    def load_obsSF(self, obsSF_path):

        # Once Colour Magnitude selection functions have been created
        # Unpickle colour-magnitude interpolants
        print("Unpickling colour-magnitude interpolant dictionaries...")
        with open(obsSF_path, "rb") as f:
            obsSF_dicts = pickle.load(f)
        print("...done.\n")

        # Initialise dictionary
        self.obsSF = SFInstanceClasses.obsSF_dicttoclass(obsSF_dicts)

    def gen_intSF(self, intSF_path, IsoCalculator):

        self.IsoCalculator = IsoCalculator

        print('Creating distance-age-metallicity interpolants...')
         #surveysf, agerng, mhrng, srng = self.createDistMhAgeInterp()
        self.instanceSF, self.instanceIMFSF, agerng, mhrng, magrng, colrng = self.createDistMhAgeInterp()
        print("...done.\n")
        # Save variable ranges into varrngs attribute
        self.varrngs.update({'agerng':agerng, 'mhrng':mhrng, 'magrng':magrng, 'colrng':colrng})

        # Convert classes to dictionaries of attributes
        instanceSF_dict = vars(self.instanceSF)
        instanceIMFSF_dict = vars(self.instanceIMFSF)

        # Decide whether to save pickled instance
        save_bool = None
        while not save_bool in ('y','n'):
            if os.path.exists(intSF_path): save_bool = input("Overwrite %s? (y/n)" % intSF_path)
            else: save_bool = input("Save as %s? (y/n)" % intSF_path)

        # Save pickled instance
        if save_bool=='y':
            print('\nPickling intrinsic SF...')
            with open(intSF_path, 'wb') as handle:
                    pickle.dump((instanceSF_dict, instanceIMFSF_dict,
                                agerng, mhrng, magrng, colrng), handle, protocol=2)

        # Can't set cm_limits because they use absolute magnitude
        self.cm_limits = None
        #self.cm_limits = (magrng[0], magrng[1], colrng[0], colrng[1])

    def gen_obsSF(self, obsSF_path):

        print('Creating colour-magnitude field interpolants...')
        obsSF_dicts = self.iterateAllFields()
        print("...done\n.")

        # Decide whether to save pickled instance
        save_bool = None
        while not save_bool in ('y','n'):
            if os.path.exists(obsSF_path): save_bool = input("Overwrite %s? (y/n)" % obsSF_path)
            else: save_bool = input("Save as %s? (y/n)" % obsSF_path)

        # Save pickled instance
        if save_bool=='y':
            print('\nPickling observable SF...')
            with open(obsSF_path, 'wb') as handle:
                pickle.dump(obsSF_dicts, handle, protocol=2)

        # Initialise dictionary
        self.obsSF = SFInstanceClasses.obsSF_dicttoclass(obsSF_dicts)

    def save_intSF(self, intSFpath):

        # Convert classes to dictionaries of attributes
        instanceSF_dict = vars(self.instanceSF)
        instanceIMFSF_dict = vars(self.instanceIMFSF)
        agerng, mhrng, magrng, colrng = \
        varrngs['agerng'], varrngs['mhrng'], varrngs['magrng'], varrngs['colrng']

        # Decide whether to save pickled instance
        save_bool = None
        while not save_bool in ('y','n'):
            if os.path.exists(obsSF_path): save_bool = input("Overwrite %s? (y/n)" % intSF_path)
            else: save_bool = input("Save as %s? (y/n)" % intSF_path)

        if save_bool=='y':
            with open(intSF_path, 'wb') as handle:
                    pickle.dump((instanceSF_dict, instanceIMFSF_dict,
                                agerng, mhrng, magrng, colrng), handle, protocol=2)

    def save_obsSF(self, obsSF_path):

        # Retrieve dictionary from class instances
        obsSF_dicts = SFInstanceClasses.obsSF_classtodict(self.obsSF)

        # Decide whether to save pickled instance
        save_bool = None
        while not save_bool in ('y','n'):
            if os.path.exists(obsSF_path): save_bool = input("Overwrite %s? (y/n)" % obsSF_path)
            else: save_bool = input("Save as %s? (y/n)" % obsSF_path)

        # Save pickled instance
        if save_bool=='y':
            print('\nPickling observable SF...')
            with open(obsSF_path, 'wb') as handle:
                pickle.dump(obsSF_dicts, handle, protocol=2)

    def combine(self, SF_inst):
        # Function currently being developed.
        # The process has more subtleties than I'd initially thought.

        SF_combined = SFGenerator(self.get_spectro, self.get_photo, self.pointings)

        # 1) Concatenate pointings from SF_inst with SF_combined.pointings
        # Add half_angle = inf entry for all-sky surveys

        # 2) Concatenate SF_inst.obsSF dictionary with self.obsSF

        # 3) __call__ needs to be capable of calling different colours and magnitudes
        # e.g. APOGEE might need H,J-Ks whilst RVS needs Grp, Gbp-Grp

        # 4) Need to convert all field IDs to strings and append with a survey identifier.`

        return SF_combined

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
            self.cm_limits

        Returns
        -------
            obsSelectionFunction - dict
                Dictionary of selection function classes for observed coordinates

        '''
        start = time.time()

        # List of fields in pointings database
        field_list = self.pointings.fieldID.values.tolist()
        # Locations for storage of solutions
        obsSF_dicts = {}

        if self.ncores>1:
            # Iteration number for each field
            field_number = np.arange(len(field_list)) + 1

            # Create a list of spectroscopic catalogue points on each field
            spectro_points_lst=[]
            for field in field_list:
                # Select survey stars
                spectro_points = self.get_spectro(field)
                spectro_points_lst.append(spectro_points)

            # zip field name, iteration number and spectroscopic points together
            field_iter = zip(field_list, field_number, spectro_points_lst)

            # Build results into a full list of multiprocessing instances
            print("multiprocessing - observable SF calculation - running on %d cores..." % self.ncores)

            # multiprocessing needs to run through an external class, otherwise it's not happy.
            # Initialise class
            iter_inst = multiprocessObsSF(self.get_spectro, self.get_photo, self.pointings,
                                        cm_limits=self.cm_limits, spectro_model=self.spectro_model, photo_model=self.photo_model,
                                        tstart=start, fieldL=len(field_list))
            # Create processor pools for multiprocessing

            pool = multiprocessing.Pool( self.ncores )
            # Run class obsMultiFunction.__call___ as external function for each field
            results = pool.map(iter_inst, field_iter, chunksize=1)

            # Locations for storage of solutions
            for r in results:
                obsSF_field, field = r
                obsSF_dicts[field] = vars(obsSF_field)

        else:
            # Field numbering to show progress
            fieldN = 0
            fieldL = len(field_list)
            tnow   = 0
            tleft  = 0
            for field in field_list:

                fieldN+=1
                sys.stdout.write("\rFinished col-mag calculation: %s, %d/%d, Time: %.2fm, Left: %.2fm" % (str(field), fieldN, fieldL, int(tnow), int(tleft)))
                sys.stdout.flush()

                # Select preferred survey stars
                #spectro_points = self.get_spectro(field)

                obsSF_field, field = _iterateField(self.get_spectro, self.get_photo, field, self.pointings.loc[field],
                                                cm_limits=self.cm_limits, spectro_model=self.spectro_model, photo_model=self.photo_model)
                obsSF_dicts[field] = vars(obsSF_field)

                tnow = (time.time() - start)/60.
                tleft = tnow*(float(fieldL)/fieldN - 1.)

        return obsSF_dicts

    def debug_mode(self, field, part='df'):

        spectro_points, photo_points, mag_range, col_range, prior_sfBounds = \
                cleanPoints(self.get_spectro, self.get_photo, field, self.pointings.loc[field], self.cm_limits)
        scales = [np.mean(spectro_points.appMag), np.mean(spectro_points.Colour),
                  np.std(spectro_points.appMag), np.std(spectro_points.Colour)]

        if part=='df':
            DF_model = StatisticalModels.BGM_TNC(x=photo_points.appMag, y=photo_points.Colour, rngx=mag_range, rngy=col_range,
                                                    runscaling=True, scales=scales, runningL=True)
            result = DF_model.optimizeParams()
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')
            SF_model = StatisticalModels.FlatRegion()

        elif part=='sf':
            DF_model = self.obsSF[field].DF_model
            SF_model = StatisticalModels.BGM_TNC(x=spectro_points.appMag, y=spectro_points.Colour, rngx=mag_range, rngy=col_range,
                                                runscaling=True, scales=scales, runningL=True,
                                                priorDF=True, photoDF=DF_model, prior_sfBounds=prior_sfBounds)
            result = SF_model.optimizeParams()
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')

        # Store information inside an SFInstanceClasses.observableSF instance where the selection function is calculated.
        instanceSF = SFInstanceClasses.observableSF(field)
        SFInstanceClasses.setattrs(instanceSF,
                                    DF_model = vars(DF_model),
                                    DF_magrange = mag_range,
                                    DF_colrange = col_range,
                                    SF_model = vars(SF_model),
                                    SF_magrange = mag_range,
                                    SF_colrange = col_range)

        self.obsSF_dicts[field] = vars(instanceSF)
        # Initialise dictionary
        self.obsSF = SFInstanceClasses.obsSF_dicttoclass(self.obsSF_dicts)

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

        agerng = self.IsoCalculator.agerng
        mhrng = self.IsoCalculator.mhrng
        # This is the ABSOLUTE magnitude range
        magrng = self.IsoCalculator.magrng
        colrng = self.IsoCalculator.colrng

        # Instance of class for creating mass dependent selection functions
        intrinsicSF = SFInstanceClasses.intrinsicSF()
        SFInstanceClasses.setattrs(intrinsicSF, IsoCalculator=self.IsoCalculator)
        # Instance of class for creating mass independent selection functions
        intrinsicIMFSF = SFInstanceClasses.intrinsicIMFSF()
        SFInstanceClasses.setattrs(intrinsicIMFSF, IsoCalculator = self.IsoCalculator)

        return intrinsicSF, intrinsicIMFSF, agerng, mhrng, magrng, colrng

    def PointsToPointings(self, stars, Phi='phi', Th='theta'):

        '''
        PointsToPointings - Generates a list for each point for all field pointings
                            for which the point's angle coordinates fall within the
                            solid angle extent of the field.

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
        Nsample = FieldAssignment.iterLimit(len(self.pointings), ncores=self.ncores)

        if self.ncores == 1:
            df = AM.AnglePointsToPointingsMatrix(stars, self.pointings,
                                              Phi, Th, 'halfangle', IDtype = self.fieldlabel_type,
                                              Nsample=Nsample)
        else:
            # List of subsections of pointings to use in parallel
            df_lst = []
            npoints = int(stars/self.ncores)
            remainder = stars%self.ncores
            for i in range(self.ncores):
                if i<remainder: dfi = stars.iloc[i*(npoints+1):(i+1)*(npoints+1)]
                else: dfi = stars.iloc[i*npoints + remainder: (i+1)*npoints + remainder]
                dfs_lst.append(dfi)

            kwargs = {'pointings':self.pointings, 'Phi':Phi, 'Th':Th, 'halfangle':'halfangle', 'IDtype':IDtype,
                     'Nsample':Nsample}
            func = AM.APTPM_parallel(**kwargs)

            # Build results into a full list of multiprocessing instances
            print("multiprocessing - field assigment - running on %d cores..." % self.ncores)

            # Create processor pools for multiprocessing
            with multiprocessing.Pool( self.ncores ) as pool:
                results = pool.map(func, df_lst)
            # Join all dataframes together
            df = pd.DataFrame()
            for r in results:
                df = pd.concat((df, r))

        return df

class multiprocessObsSF():

    '''
    multiprocessObsSF - Class for calculating observable selection function
        - This allows us to run each field in a pool of parallel cores.

    Parameters
    ----------
        get_spectro
        get_photo
        pointings: pandas dataframe
            - Field location pointings

    **kwargs
    --------
        cm_limits=None: 4-tuple of float
            - Limits on colour and magnitude to act as boundaries if field boundaries not given
        spectro_model=None: tuple
            - Model to be use for selection function
        photo_model=None: tuple
            - Mudel to be used for distribution function
        tstart=0: float
            - time.time() when observable SF calculation was started
        fieldN=0: int
            - Number of fields in calculation
    '''

    def __init__(self, get_spectro, get_photo, pointings,
                       cm_limits=None, spectro_model=None, photo_model=None,
                       tstart=0, fieldL=0):

        self.get_spectro = get_spectro
        self.get_photo = get_photo
        # Dataframe of field pointings
        self.pointings = pointings
        # Colour and magnitude limits of survey
        self.cm_limits = cm_limits
        # Model for selection function (e.g. ('GMM', 1))
        self.spectro_model = spectro_model
        # Model for distribution function (e.g. ('GMM', 2))
        self.photo_model = photo_model

        # Start time for observable SF calculation
        self.tstart=tstart
        # Number of fields to iterate
        self.fieldL=fieldL

    def __call__(self, field_iter):

        '''
        __call__ - Run the observable SF calculation for the given field

        Parameters
        ----------
            field_iter: tuple
                - fieldID, iteration number and spectroscopic stars of field
        Returns
        -------
            ans: tuple
                - Values returned from iterateField
        '''

        field, fieldN, spectro_points = field_iter
        fieldL = self.fieldL

        ans = iterateField(self.get_spectro, self.get_photo, field, self.pointings.loc[field],
                            cm_limits=self.cm_limits, spectro_model=self.spectro_model, photo_model=self.photo_model)

        # Time taken to get to this point
        tnow = (time.time() - self.tstart)/60.
        tleft = tnow*(float(fieldL)/fieldN - 1.)
        # output progress to stdout
        sys.stdout.write("\rcol-mag calculation finished for field: %s, %d/%d, Time: %dm, Left: %dm" \
                        % (str(field), fieldN, fieldL, int(tnow), int(tleft)))
        sys.stdout.flush()

        return ans

def _iterateField(get_spectro, get_photo, field, fieldpointing,
                    cm_limits=None, spectro_model=('GMM', 2), photo_model=('GMM', 3)):

    spectro_points, photo_points, mag_range, col_range, prior_sfBounds = \
            cleanPoints(get_spectro, get_photo, field, fieldpointing, cm_limits)

    # Only create an interpolant if there are any points in the region
    if len(spectro_points)>0:

        scales = [np.mean(spectro_points.appMag), np.mean(spectro_points.Colour),
                  np.std(spectro_points.appMag), np.std(spectro_points.Colour)]

        if spectro_model[0]=='GMM':

            # Number of Gaussian components
            nComponents = modelinfo[1]

            DF_model = StatisticalModels.GaussianEM(x=photo_points.appMag, y=photo_points.Colour, nComponents=photo_model[1],
                                                rngx=mag_range, rngy=col_range, runscaling=True, runningL=False)
            result = DF_model.optimizeParams(method='Powell', init='kmeans')
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')

            SF_model = StatisticalModels.GaussianEM(x=spectro_points.appMag, y=spectro_points.Colour, nComponents=spectro_model[1],
                                                rngx=mag_range, rngy=col_range, runscaling=True, runningL=False,
                                                priorDF=True, photoDF=DF_model)
            result = SF_model.optimizeParams(method='Powell', init='kmeans')
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')

        elif spectro_model[0]=='BGM_TNC':

            DF_model = StatisticalModels.BGM_TNC(x=photo_points.appMag, y=photo_points.Colour, rngx=mag_range, rngy=col_range,
                                                    runscaling=True, scales=scales, runningL=True)
            result = DF_model.optimizeParams()
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')

            SF_model = StatisticalModels.BGM_TNC(x=spectro_points.appMag, y=spectro_points.Colour, rngx=mag_range, rngy=col_range,
                                                runscaling=True, scales=scales, runningL=True,
                                                priorDF=True, photoDF=DF_model, prior_sfBounds=prior_sfBounds)
            result = SF_model.optimizeParams()
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')

        elif spectro_model[0]=='Flat':

            DF_model = StatisticalModels.Empty()
            SF_model = StatisticalModels.FlatRegion(len(points), mag_range, col_range)

        else: raise ValueError('Model not recognised.')

    else:
        # There are no stars on the field plate
        instanceSF = SFInstanceClasses.observableSF(field)
        # Create class instances so that the models will still work.
        DF_model = StatisticalModels.Empty()
        SF_model = StatisticalModels.Empty()

    # Store information inside an SFInstanceClasses.observableSF instance where the selection function is calculated.
    instanceSF = SFInstanceClasses.observableSF(field)
    SFInstanceClasses.setattrs(instanceSF,
                                DF_model = vars(DF_model),
                                DF_magrange = mag_range,
                                DF_colrange = col_range,
                                SF_model = vars(SF_model),
                                SF_magrange = mag_range,
                                SF_colrange = col_range)

    return instanceSF, field

def cleanPoints(get_spectro, get_photo, field, fieldpointing, cm_limits):

    spectro_points = get_spectro(field)
    spectro_points = pd.DataFrame(np.array(spectro_points), columns=['appMag', 'Colour'])

    photo_points = get_photo(field)
    photo_points = pd.DataFrame(np.array(photo_points), columns=['appMag', 'Colour'])

    # Only create an interpolant if there are any points in the region
    if len(spectro_points)>0:

        prior_sfBounds = np.zeros((2,2))

        # Use given limits to determine boundaries of dataset
        # apparent mag upper bound
        if fieldpointing.Magmin == "NoLimit":
            if cm_limits is None: mag_min = np.min(spectro_points.appMag) - 1
            else: mag_min = cm_limits[0]
            prior_sfBounds[0,0] = np.min(photo_points.appMag)
        else:
            mag_min = fieldpointing.Magmin - 1
            prior_sfBounds[0,0] = mag_min
        # apparent mag lower bound
        if fieldpointing.Magmax == "NoLimit":
            if cm_limits is None: mag_max = np.max(spectro_points.appMag) + 1
            else: mag_max = cm_limits[1]
            prior_sfBounds[0,1] = np.max(photo_points.appMag)
        else:
            mag_max = fieldpointing.Magmax + 1
            prior_sfBounds[0,1] = mag_max
        # colour uppper bound
        if fieldpointing.Colmin == "NoLimit":
            if cm_limits is None: col_min = np.min(spectro_points.Colour) - 0.1
            else: col_min = cm_limits[2]
            prior_sfBounds[1,0] = np.min(photo_points.Colour)
        else:
            col_min = fieldpointing.Colmin - 0.1
            prior_sfBounds[1,0] = col_min
        # colour lower bound
        if fieldpointing.Colmax == "NoLimit":
            if cm_limits is None: col_max = np.max(spectro_points.Colour) + 0.1
            else: col_max = cm_limits[3]
            prior_sfBounds[1,1] = np.max(photo_points.Colour)
        else:
            col_max = fieldpointing.Colmax + 0.1
            prior_sfBounds[1,1] = col_max

        # Chose only photometric survey points within the colour-magnitude region.
        photo_points = photo_points[(photo_points.appMag >= mag_min)&\
                                    (photo_points.appMag <= mag_max)&\
                                    (photo_points.Colour >= col_min)&\
                                    (photo_points.Colour <= col_max)]
        # If spectro points haven't been chosen from the full region, limit to this subset.
        spectro_points = spectro_points[(spectro_points.appMag >= mag_min)&\
                                    (spectro_points.appMag <= mag_max)&\
                                    (spectro_points.Colour >= col_min)&\
                                    (spectro_points.Colour <= col_max)]

    else:
        mag_min, mag_max, col_min, col_max = None, None, None, None
        prior_sfBounds = None
        print("No stars on field")

    return spectro_points, photo_points, (mag_min, mag_max), (col_min, col_max), prior_sfBounds

def iterateField(get_spectro, get_photo, field, fieldpointing,
                cm_limits=None, spectro_model=('GMM', 1), photo_model=('GMM', 2)):

    '''
    iterateField- Generates a selection function instance for a given field

    Parameters
    ----------
        get_spectro
        get_photo
        field: field label type (str, int, float, np.float64...)
            The field being iterated
        fieldpointing:

    kwargs
    ------
        cm_limits=None: 4-tuple of float
            Limits on colour and magnitude to act as boundaries if field boundaries not given
        spectro_model=('GMM', 1): tuple
            - Model to be use for selection function
        photo_model=('GMM', 2): tuple
            - Mudel to be used for distribution function
    Returns
    -------
        instanceSF - SFInstanceClasses class
            Class for calculating the selection function value given colour and magnitude
            for a field

        field - field type (str, int, float, np.float64...)
            The field in question

    '''

    spectro_points = get_spectro(field)
    spectro_points = pd.DataFrame(np.array(spectro_points), columns=['appMag', 'Colour'])

    photo_points = get_photo(field)
    photo_points = pd.DataFrame(np.array(photo_points), columns=['appMag', 'Colour'])

    # Only create an interpolant if there are any points in the region
    if len(spectro_points)>0:

        prior_sfBounds = np.zeros((2,2))

        # Use given limits to determine boundaries of dataset
        # apparent mag upper bound
        if fieldpointing.Magmin == "NoLimit":
            if cm_limits is None: mag_min = np.min(spectro_points.appMag) - 1
            else: mag_min = cm_limits[0]
            prior_sfBounds[0,0] = np.min(photo_points.appMag)
        else:
            mag_min = fieldpointing.Magmin - 1
            prior_sfBounds[0,0] = mag_min
        # apparent mag lower bound
        if fieldpointing.Magmax == "NoLimit":
            if cm_limits is None: mag_max = np.max(spectro_points.appMag) + 1
            else: mag_max = cm_limits[1]
            prior_sfBounds[0,1] = np.max(photo_points.appMag)
        else:
            mag_max = fieldpointing.Magmax + 1
            prior_sfBounds[0,1] = mag_max
        # colour uppper bound
        if fieldpointing.Colmin == "NoLimit":
            if cm_limits is None: col_min = np.min(spectro_points.Colour) - 0.1
            else: col_min = cm_limits[2]
            prior_sfBounds[1,0] = np.min(photo_points.Colour)
        else:
            col_min = fieldpointing.Colmin - 0.1
            prior_sfBounds[1,0] = col_min
        # colour lower bound
        if fieldpointing.Colmax == "NoLimit":
            if cm_limits is None: col_max = np.max(spectro_points.Colour) + 0.1
            else: col_max = cm_limits[3]
            prior_sfBounds[1,1] = np.max(photo_points.Colour)
        else:
            col_max = fieldpointing.Colmax + 0.1
            prior_sfBounds[1,1] = col_max

        # Chose only photometric survey points within the colour-magnitude region.
        photo_points = photo_points[(photo_points.appMag >= mag_min)&\
                                    (photo_points.appMag <= mag_max)&\
                                    (photo_points.Colour >= col_min)&\
                                    (photo_points.Colour <= col_max)]
        # If spectro points haven't been chosen from the full region, limit to this subset.
        spectro_points = spectro_points[(spectro_points.appMag >= mag_min)&\
                                    (spectro_points.appMag <= mag_max)&\
                                    (spectro_points.Colour >= col_min)&\
                                    (spectro_points.Colour <= col_max)]

        scales = [np.mean(spectro_points.appMag), np.mean(spectro_points.Colour),
                  np.std(spectro_points.appMag), np.std(spectro_points.Colour)]

        # Interpolate for photo data - Calculates the distribution function
        DF_model, DF_magrange, DF_colrange = CreateInterpolant(photo_points,
                                                             (mag_min, mag_max), (col_min, col_max),
                                                             range_limited=True, scales=scales,
                                                             datatype = "photo", modelinfo=photo_model)
        # Interpolate for spectro data - Calculates the selection function
        SF_model, SF_magrange, SF_colrange = CreateInterpolant(spectro_points,
                                                              (mag_min, mag_max), (col_min, col_max),
                                                              scales=scales,
                                                              datatype = "spectro", photoDF=DF_model, prior_sfBounds=prior_sfBounds,
                                                              modelinfo=spectro_model)

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

def CreateInterpolant(points, mag_range, col_range,
                      Grid = False, range_limited=False, scales=None,
                      datatype="", photoDF=None, prior_sfBounds=None,
                      modelinfo=('GMM', 1)):

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
                                        'appMag', 'Colour', scales=scales,
                                        datatype=datatype, modelinfo=modelinfo)
        elif datatype == "spectro":
            model = PoissonLikelihood(points, mag_range, col_range,
                                        'appMag', 'Colour', scales=scales,
                                        datatype=datatype, photoDF=photoDF, prior_sfBounds=prior_sfBounds,
                                        modelinfo=modelinfo)

        return model, mag_range, col_range

    # Ratio of number of stars in the field.
    elif Process == "Number":

        model = StatisticalModels.FlatRegion(len(points), mag_range, col_range)

        return model, mag_range, col_range

# Used when Process == Poisson
def PoissonLikelihood(points,
                     mag_range, col_range,
                     mag_label, col_label, scales=None,
                     datatype="", photoDF=None, prior_sfBounds=None,
                     modelinfo=('GMM', 1)):
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
        if datatype=='photo':
            model = StatisticalModels.GaussianEM(x=x, y=y, nComponents=nComponents,
                                                rngx=mag_range, rngy=col_range, runscaling=True)
            model.runningL = False
            result = model.optimizeParams(method='Powell', init='kmeans')
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')
        # Add in SFxDF<DF constraint for the spectrograph distribution
        elif datatype == "spectro":
            model = StatisticalModels.GaussianEM(x=x, y=y, nComponents=nComponents,
                                                rngx=mag_range, rngy=col_range, runscaling=True,
                                                priorDF=True, photoDF=photoDF)
            model.runningL = False
            result = model.optimizeParams(method='Powell', init='kmeans')
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')
            #model.photoDF, model.priorDF = (photoDF, True)
        else: raise ValueError('Datatype not understood: %s' % datatype)
        # Test integral if you want to see the value/error in the integral when calculated
        # model.testIntegral()
    if modelType == 'BGM_TNC':

        # Generate the model
        if datatype=='photo':
            model = StatisticalModels.BGM_TNC(x=x, y=y, rngx=mag_range, rngy=col_range,
                                            runscaling=True, scales=scales)
            model.runningL = True
            result = model.optimizeParams()
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')
        elif datatype == "spectro":
            model = StatisticalModels.BGM_TNC(x=x, y=y, rngx=mag_range, rngy=col_range, runscaling=True, scales=scales,
                                                priorDF=True, photoDF=photoDF, prior_sfBounds=prior_sfBounds)
            model.runningL = True
            result = model.optimizeParams()
            #result_emcee = model.optimizeParams(method='emceeBall', init='reset')
        else: raise ValueError('Datatype not understood: %s' % datatype)

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

class external():
    '''
    external - Allows multiprocessing from inside a class

    Parameters
    ----------
        func: function
            - The function being iterated over in parallel
        args: tuple
            - All stationary arguments of function
        kwargs: dict
            - All stationary kwarguments of function
    '''

    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, moreargs):

        '''
        __call__ - Run the function on the given arguments

        Parameters
        ----------
            moreargs: tuple
                - Arguments which change with each iteration
        Returns
        -------
            ans: output of function
                - The value returned by running the function
        '''

        args = moreargs+self.args
        ans = self.func(*args, **self.kwargs)
        return ans
