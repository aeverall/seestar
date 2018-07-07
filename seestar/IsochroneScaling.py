'''
IsochroneScaling - Module for calculation of colours and magnitudes of stars
                   given intrinsic properties of stars (age, metallicity, mass, distance)
                   using the stellar Isochrones.

Classes
-------
IntrinsicToObservable - Class for creating a calculator which can determine the colours
                        absolute and apparent magnitudes of stars given their intrinsic
                        properties.

Functions
---------


Requirements
------------

ArrayMechanics.py
'''

import dill, sys, pickle
import numpy as np
import pandas as pd
from itertools import  product

import scipy
from scipy.interpolate import RegularGridInterpolator as RGI

from seestar import ArrayMechanics as AM

class IntrinsicToObservable():

    '''
    IntrinsicToObservable - Class for creating a calculator which can determine the colours
                            absolute and apparent magnitudes of stars given their intrinsic
                            properties.

    Parameters
    ----------
        age - array or float (same size array as mh, mass, s) (Gyr)
            ages of objects to be calculated
        mh - array or float (dex)
            metallicity of objects to be calculated
        mass - array or float (solar mass)
            mass of objects to be calculated
        s - array or float (kpc)
            distance of objects to be calculated

    Functions
    ---------
        __init__

        __call__ - Runs ColourMapp
                - Calculates the colour and apparent magnitude of objects
                    given their age, metallicity, mass and distance.

        CreateFromIsochrones - Generates a scaled mass regime, colour and magnitude
                            interpolants from the raw stellar isochrones chosen.

        pickleColMag - Save pickle files of the colour and magnitude interpolants

        pickleMagnitudes - Save pickle files of the magnitude interpolants

    Returns
    -------
        Colour - array or float
            Colour of objects being calculated in bands A-B
        Mapp - array or float
            Apparent magnitude of objects being calculated in band C

    Dependencies
    ------------
        dill, pickle
    '''
    
    def __init__(self):
    
        # Isochrone file location
        self.iso_pickle = ''
        # Storage of isochrone information
        self.isoage, self.isomh, self.isodict = (None, None, None)
    
        # Storage of scaled mass values
        self.m_scaled = None

        # Columns for A, B and C magnitudes in iso_pickle file
        # Colour = A-B
        self.columnABC = (13, 15, 14)
        # Column for initial mass in iso_pickle file
        self.columnMi = 2
        
        # Maximum age and metallicity ranges
        self.agerng = (0, 13)
        self.mhrng = (-2.5, 0.5)

        # magnitude and colour ranges
        self.magrng = None
        self.colrng = None

    def __call__(self, age, mh, mass, s):

        '''
        __call__ - Runs ColourMapp
                - Calculates the colour and apparent magnitude of objects
                    given their age, metallicity, mass and distance.

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated
            s - array or float (kpc)
                distance of objects to be calculated

        Returns
        -------
            Colour - array or float
                Colour of objects being calculated in bands A-B
            Mapp - array or float
                Apparent magnitude of objects being calculated in band C
        '''

        Colour, Mapp = self.ColourMapp(age, mh, mass, s)

        return Colour, Mapp
    
    def CreateFromIsochrones(self, file):

        '''
        CreateFromIsochrones - Generates a scaled mass regime, colour and magnitude
                            interpolants from the raw stellar isochrones chosen.

        Parameters
        ----------
            None

        Returns
        -------
            None

        Inherited
        ---------
            self.iso_pickle - str
                    File containing the isochrones

        Bequeathed
        ----------
            self.isoage - 1D array
                    - Ages used in isochrones

            self.isomh - 1D array
                    - metallicities used in isochrones

            self.isodict - dictionary
                    - Contains all isochrone arrays

            self.Mmin_interp - RegularGridInterpolator
                    - Minimum mass interpolants over isochrones ((age, mh))
            self.Mmax_interp - RegularGridInterpolator
                    - Maximum mass interpolants over isochrones ((age, mh))

            self.m_scaled - 1D array
                    - scaled masses used in interpolation grid

            self.iso_info - dict
                Dictionary of isochrones after being calculated for scaled mass distribution.

            self.magAinterp, self.magBinterp, self.magCinterp - RegularGridInterpolator
                    - Absolute magnitude interpolants for each magnitude band ((age, mh, scaled mass))
                    - Interpolants over grid of: isoage, isomh, m_scaled
        '''
        
        # Import isochrone dictionaries and generate isoage, isomh and isodict entries
        isoage, isomh, isodict = ImportIsochrones(file)
        self.isoage, self.isomh, self.isodict = isoage, isomh, isodict
        
        # Calculate interpolant of min and max mass values from isochrones
        Mmin_interp, Mmax_interp = isoMaxMass(isoage, isomh, isodict)
        self.Mmin_interp = Mmin_interp
        self.Mmax_interp = Mmax_interp        

        # Determine the optimum distribution of scaled masses
        m_scaled = ScaledMasses(isoage, isomh, isodict, self.scaling,
                                columnABC=self.columnABC, columnMi=self.columnMi)
        self.m_scaled = m_scaled        
        
        # Create new isochrone dictionary with scaled mass distribution
        iso_info = Redistribution(isoage, isomh, isodict, m_scaled, self.scaling, self.unscaling,
                                columnABC=self.columnABC, columnMi=self.columnMi)
        self.iso_info = iso_info
        
        # Calculate colour and magnitude interpolants in terms of age, mh, mass scaled
        magA_interp, magB_interp, magC_interp, col_interp, agerng, mhrng, magrng, colrng = \
                cmInterpolation(isoage, isomh, iso_info, m_scaled, agerng=self.agerng, mhrng=self.mhrng)
        self.magA_interp = magA_interp
        self.magB_interp = magB_interp
        self.magC_interp = magC_interp
        self.col_interp = col_interp

        # update age and metallicity ranges
        self.agerng = agerng
        self.mhrng = mhrng
        # Return the colour and magnitude ranges
        self.magrng = magrng
        self.colrng = colrng
    
    def pickleColMag(self, pickle_path):

        '''
        pickleColMag - Save pickle files of the colour and magnitude interpolants

        Parameters
        ----------
            pickle_path - str
                File location for saving pickled instance

        Returns
        -------
            None
        '''
        
        with open(pickle_path, 'wb') as handle:
            pickle.dump((self.Mmax_interp, self.Mmin_interp, 
                         self.col_interp, self.magA_interp, self.magB_interp, self.magC_interp,
                         self.magrng, self.colrng), handle)

    def pickleMagnitudes(self, pickle_path):

        '''
        pickleMagnitudes - Save pickle files of the magnitude interpolants

        Parameters
        ----------
            pickle_path - str
                File location for saving pickled instance

        Returns
        -------
            None
        '''
        
        with open(pickle_path, 'wb') as handle:
            pickle.dump((self.Mmax_interp, self.Mmin_interp, 
                         self.magA_interp, self.magB_interp, self.magC_interp), handle)
            
    def LoadColMag(self, pickle_path):

        '''
        LoadColMag - Load in colour-magnitude pickled interpolants

        Parameters
        ----------
            pickle_path: str
                - Path to pickle file of colour-magnitude interpolants

        Returns
        -------
            None
        '''
        
        with open(pickle_path, "rb") as input:
            self.Mmax_interp, self.Mmin_interp, \
            self.col_interp, self.magA_interp, self.magB_interp, self.magC_interp, \
            self.magrng, self.colrng = pickle.load(input)

    def LoadMagnitudes(self, pickle_path):

        '''
        LoadColMag - Load in three magnitude interpolant instances.

        Parameters
        ----------
            pickle_path: str
                - Path to pickle file of magnitude interpolants
                
        Returns
        -------
            None
        '''
        
        with open(pickle_path, "rb") as input:
            Mmax_interp, Mmin_interp, magA_interp, magB_interp, magC_interp = pickle.load(input)
            
        self.Mmin_interp = Mmin_interp
        self.Mmax_interp = Mmax_interp
        self.magA_interp = magA_interp
        self.magB_interp = magB_interp
        self.magC_interp = magC_interp

    
    def massScaled(self, age, mh, mass):

        '''
        massScaled - calculate scaled mass from mass using isochrone limits

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated

        Inherited
        ---------
            Mmax_interp, Mmin_interp: RGI
                - Interpolant over age and metallicity of the maximum/minimum masses in isochrones
            scaling: lambda
                - Function for calculating scaled mass from age, metallicity and mass

        Returns
        -------
            m_scaled: arr of float
                - scaled mass of all stars in the array
        '''
        
        # Find the max/min mass from isochrones of stars with age and metallicity.
        Mmax = self.Mmax_interp((age, mh))
        Mmin = self.Mmin_interp((age, mh))
        
        # Convert mass to scaled mass for each isochone(interpolated between)
        m_scaled = self.scaling(mass, Mmax, Mmin)
        
        return m_scaled
    
    def ColourMabs(self, age, mh, mass):

        '''
        ColourMabs - Get colour and absolute magnitude from age, metallicity and mass

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated

        Inherited
        ---------
            m_scaled: arr of float
                - scaled mass of all stars in the array
            col_interp: RGI
                - Interpolant over age, metallicity and scaled mass of the colour
            magC_interp: RGI
                - Interpolant over age, metallicity and scaled mass of the C-band magntiude


        Returns
        -------
            Colour: arr of float
                - Determined colours of all stars in array
            Mabs: arr of float
                - Determined absolute magnitudes of all stars in array
        '''
        
        # Convert mass to scaled mass
        m_scaled = self.massScaled(age, mh, mass)
        
        #Colour = self.col_interp((age, mh, m_scaled))
        Colour = self.magA_interp((age, mh, m_scaled)) - self.magB_interp((age, mh, m_scaled))

        Mabs = self.magC_interp((age, mh, m_scaled))
        
        return Colour, Mabs
    
    def ColourMapp(self, age, mh, mass, s):

        '''
        ColourMapp - Calculates the colour and apparent magnitude of objects
                    given their age, metallicity, mass and distance.

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated
            s - array or float (kpc)
                distance of objects to be calculated

        Returns
        -------
            Colour - array or float
                Colour of objects being calculated in bands A-B
            Mapp - array or float
                Apparent magnitude of objects being calculated in band C
        '''

        Colour, Mabs = self.ColourMabs(age, mh, mass)
        s = s.values # Convert from series to array
        Mapp = self.appmag(Mabs, s)

        return Colour, Mapp

    def AbsMags(self, age, mh, mass):

        '''
        AbsMag - Determines absolute magnitude in all three bands of the stars in the
                array given their age, metallicity and mass.

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated

        Inherited
        ---------
            magA_interp, magB_interp, magC_interp: RGI
                - Interpolant over age, metallicity and scaled mass of the C-band magntiude

        Returns
        -------
            Aabs, Babs, Cabs: arr of float
                - Absolute magnitudes of all stars in array in each of the bands.
        '''

        # Convert mass to scaled mass
        m_scaled = self.massScaled(age, mh, mass)

        Aabs = self.magA_interp((age, mh, m_scaled))
        Babs = self.magB_interp((age, mh, m_scaled))
        Cabs = self.magC_interp((age, mh, m_scaled))

        return Aabs, Babs, Cabs

    def AppMags(self, age, mh, mass, s):

        '''
        AbsMag - Determines apparent magnitude in all three bands of the stars in the
                array given their age, metallicity and mass.

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated
            s - array or float (kpc)
                distance of objects to be calculated

        Inherited
        ---------
            appmag: lambda
                - Calculate apparent magnitude from absolute magnitude and distance.

        Returns
        -------
            Aapp, Bapp, Capp: arr of float
                - Apparent magnitudes for all stars in array in each band.
        '''

        # Convert mass to scaled mass
        Aabs, Babs, Cabs = self.AbsMags(age, mh, mass)

        Aapp = self.appmag(Aabs, s)
        Bapp = self.appmag(Babs, s)
        Capp = self.appmag(Cabs, s)

        return Aapp, Bapp, Capp

    def appmag(self, absmag, s):
        """ Conversion of absolute to apparent magnitude """
        return absmag + 5*np.log10(s*1000/10)
    def absmag(self, appmag, s):
        """ Conversion of apparent to absolute magnitude """
        return appmag - 5*np.log10(s*1000/10)

    def scaling(self, mass, Mmax, Mmin):
        """ Mass scaling relation """
        return (mass-Mmin)/(Mmax-Mmin)
    def unscaling(self, mass_scaled, Mmax, Mmin):
        """ Mass unscaling relation """
        return mass_scaled*(Mmax-Mmin) + Mmin



def ImportIsochrones(iso_pickle):

    '''
    ImportIsochrones - Loads in dill file of isochrone information.
        - Could potentially generalise this to more data formats

    Parameters
    ----------
        iso_pickle: str
            - path to dill file of isochrones

    Returns
    -------
        isoage, isomh: arr of float
            - Ages and metallicities of isochrones in the database
        isodict: dict
            - Dictionary of isochrones
    '''

    # Extract isochrones
    print(iso_pickle)
    print("Unpickling isochrone dictionaries...")
    with open(iso_pickle, "rb") as f:
        pi = pickle.load(f)
    print("...done.\n")

    isoage    = pi['isoage']
    isomh     = pi['isomh']
    isodict   = pi['isodict']
    
    return isoage, isomh, isodict

def isoMaxMass(isoage, isomh, isodict):

    '''
    isoMaxMass - Finds the dependence of max and min mass of isochrones on age and metallicity.

    Parameters
    ----------
        isoage, isomh: arr of float
            - Ages and metallicities of isochrones in the database
        isodict: dict
            - Dictionary of isochrones

    Returns
    -------
        Mmin_interp, Mmax_interp: RGI
            - Interpolant of min and max mass over age and metallicity space
    '''

    # Scaling interpolation
    agemh_scale = []
    for age in isoage:
        for mh in isomh:

            interpname = stringLength(age, mh, isodict)
            isochrone   = isodict[interpname]

            # Extract information from isochrones
            Mi = isochrone[:,2]

            # Madimum mass to be used for scaling functions
            Mmax = np.max(Mi)
            Mmin = np.min(Mi)

            agemh_scale.append([age, mh, Mmax, Mmin])

    agemh_scale = np.array(agemh_scale)
    agemh_scale = pd.DataFrame(agemh_scale, columns=['age','mh','mmax','mmin'])

    agegrid = np.array(agemh_scale.age)
    mhgrid = np.array(agemh_scale.mh)

    agemh_scale.set_index(['age','mh'], inplace=True)
    Mmax = np.array(agemh_scale['mmax'].unstack())
    Mmin = np.array(agemh_scale['mmin'].unstack())

    Mmax_interp = RGI((isoage, isomh), Mmax, bounds_error=False, fill_value=np.nan)
    Mmin_interp = RGI((isoage, isomh), Mmin, bounds_error=False, fill_value=np.nan)

    return Mmin_interp, Mmax_interp

def ScaledMasses(isoage, isomh, isodict, scaling,
                columnABC=(13, 15, 14), columnMi=2):

    '''
    ScaledMasses - Generates a set of scaled mass values which are randomly selected over a cdf
                to best sample the regions of the isochrones which change fastest.

    Parameters
    ----------
        isoage, isomh: arr of float
            - Ages and metallicities of isochrones in the database
        isodict: dict
            - Dictionary of isochrones
        scaling: lambda
            - Calculation of scaled mass

    **kwargs
    --------
        columnABC=(13,15,14): 3tuple of int
            - Columns in array (from isodict) which hold magnitudes A, B and C
        columnMi=2: int
            - Column in array (from isodict) which holds mass

    Returns
    -------
        m_scaled: arr of float
            - Set of values of mass scaled which well sample the isochrones.
    '''

    Mgrid = np.linspace(0., 1., 500)

    mvals = []

    # For logging progress
    i=0

    for age in isoage:
        for mh in isomh:

            interpname = stringLength(age, mh, isodict)
            isochrone   = isodict[interpname]

            Mi = isochrone[:,columnMi]
            Aabs = isochrone[:,columnABC[0]]
            Babs = isochrone[:,columnABC[1]]
            Cabs = isochrone[:,columnABC[2]]
            Col = Aabs - Babs

            # Same for all isochrones
            Mmin = np.min(Mi)
            Mmax = np.max(Mi)
            # Scaled mass varies from 0 to 1
            M_scaled = scaling(Mi, Mmax, Mmin)

            isointerp_MAabs = scipy.interpolate.interp1d(M_scaled, Aabs)#, bounds_error=False, fill_value=np.nan)
            isointerp_MBabs = scipy.interpolate.interp1d(M_scaled, Babs)#, bounds_error=False, fill_value=np.nan)
            isointerp_MCabs = scipy.interpolate.interp1d(M_scaled, Cabs)#, bounds_error=False, fill_value=np.nan)
            isointerp_MCol = scipy.interpolate.interp1d(M_scaled, Col)#, bounds_error=False, fill_value=np.nan)

            # Stagger m in root(col**2+mag**2) intervals
            # Starting list for m values
            mlist = []
            # Starting at minimum mass value
            m=np.min(M_scaled)
            dy = 0.1
            dm=0.001
            while (m+dm)<1.0:
                mlist.append(m)
                dColdm = np.abs(isointerp_MCol(m+dm)-isointerp_MCol(m))/dm
                dCabsdm = np.abs(isointerp_MCabs(m+dm)-isointerp_MCabs(m))/dm
                dm = dy/np.sqrt(dColdm**2+dCabsdm**2)
                m+=dm

            # Add m values to the entire list
            mvals.extend(mlist)


        # Update progress on going through age values
        sys.stdout.write("Scaled masses spacing: " + str(i)+' / '+str(len(isoage))+'\r')
        i += 1
    print("")

    mvals =np.array(mvals)


    N=1000
    val = np.linspace(0., 1., N)
    cdf = np.zeros(N)

    for i in range(N):

        cdf[i] = float(np.sum(mvals<val[i]))/len(mvals)

        # Update progress on going through age values
        sys.stdout.write("Scaled masses points: " +str(i)+' / '+str(N)+'\r')
    print("")

    cdfinterp = scipy.interpolate.interp1d(cdf, val)


    Nmass = 500

    m_scaled = cdfinterp(np.random.rand(Nmass))
    m_scaled = np.sort(m_scaled)

    return m_scaled

def Redistribution(isoage, isomh, isodict, 
                   m_scaled, scaling, unscaling,
                   columnABC=(13, 15, 14), columnMi=2):

    '''
    Redistribution - Recalculates values along the isochrones based on
                    the distribution of m_scaled values

    Parameters
    ----------
        isoage - 1D array
            Set of age values in isochrones.
        isomh  - 1D array
            Set of metallicity values in isochrones.
        isodict - dict
            Dictionary containing all isochrones.
        m_scaled - 1D array
            Set of scaled mass values to be used in new isochrone distribution.
        scaling - lambda/function instance
            Function for scaling mass (Mi, Mmax, Mmon)
        unscaling - lambda/function instance
            Function for unscaling mass (Mi, Mmax, Mmin)

    **kwargs
    --------
        columnABC - 3-tuple of ints
            Columns of magnitude bands in isochrones
        columnMi - int
            Column of initial mass in isochrones

    Returns
    -------
        iso_info - dict
            Dictionary of isochrones after being calculated for scaled mass distribution.
    '''

    iso_info = {}

    # For logging progress
    i=0
    for age in isoage:
        for mh in isomh:
            # Retrieve isochrones from pi
            interpname = stringLength(age, mh, isodict)
            isochrone   = isodict[interpname]

            # Extract information from isochrones
            Mi = isochrone[:,columnMi]
            Aabs = isochrone[:,columnABC[0]]
            Babs = isochrone[:,columnABC[1]]
            Cabs = isochrone[:,columnABC[2]]

            # Madimum mass to be used for scaling functions
            Mmax = np.max(Mi)
            Mmin = np.min(Mi)

            # Determine scaled mass
            Mi_scaled = scaling(Mi, Mmax, Mmin)

            # Create isochrone interpolants
            isointerp_MAabs = scipy.interpolate.interp1d(Mi_scaled, Aabs)#, bounds_error=False, fill_value=np.nan)
            isointerp_MBabs = scipy.interpolate.interp1d(Mi_scaled, Babs)#, bounds_error=False, fill_value=np.nan)
            isointerp_MCabs = scipy.interpolate.interp1d(Mi_scaled, Cabs)#, bounds_error=False, fill_value=np.nan)

            # Recalculate coordinates using distribution of scaled masses
            Mi = unscaling(m_scaled, Mmax, Mmin)
            Aabs = isointerp_MAabs(m_scaled)
            Babs = isointerp_MBabs(m_scaled)
            Cabs = isointerp_MCabs(m_scaled)

            # Place isochrone information in a dataframe
            isoarr = np.column_stack((Mi, Aabs, Babs, Cabs))
            isodf = pd.DataFrame(isoarr, columns=['mass_i', 'Aabs', 'Babs', 'Cabs'])

            # Write dataframe to isochrone dictionary
            iso_info[interpname] = isodf


        # Update progress on going through age values
        sys.stdout.write("Redistribution: " +str(i)+' / '+str(len(isoage))+'\r')
        i += 1

    return iso_info

def cmInterpolation(isoage, isomh, iso_info, m_scaled, 
                    agerng= (0,13), mhrng=(-2.5, 0.5)):

    '''
    cmInterpolation - Interpolate over colour and magnitude on an age-metallicity-scaled mass grid.

    Parameters
    ----------
        isoage, isomh: arr of float
            - Ages and metallicities of all isochrones in the catalogue
        iso_info: dict
            - Dictionary of all isochrones where values are "age%fmh%f" % age, metallicity
        m_scaled: arr of float
            - Array of scaled mass values to be used for interpolation points

    **kwargs
    --------
        agerng=(0,13): tuple of float
            - Min and max age to be used (will actually be a larger range as will take nearest isochrone instide)
        mhrng=(-2.5, 0.5): tuple of float
            - Min and max metallicity to be used (will actually be a larger range as will take nearest isochrone instide)

    Returns
    -------
        magA_interp, magB_interp, magC_interp: RGI
            - Interpolation of absolute magnitude over age-metallicity-scaled mass grid
        col_interp: RGI
            - Interpolation of colour over age-metallicity-scaled mass grid
        agerng, mhrng: tuple of float
            - Min and max age and metallicity taken from max and min isochrones
        magrng, colrng: tuple of float
            - Min and max magnitude and colour found from all isochrones
    '''

    # Construct mass grid
    massgrid = m_scaled

    # Grids are calculated between age and metallicity ranges
    # Construct age grid
    jagemin    = max(0, 
                     np.sum(isoage<agerng[0])-1)
    jagemax     = min(len(isoage), 
                      np.sum(isoage<agerng[1]) +1)
    agegrid = isoage[jagemin:jagemax]

    # Construct metallicity grid
    jmhmin     = max(0,
                     np.sum(isomh<mhrng[0])-1)
    jmhmax     = min(len(isomh),
                     np.sum(isomh<mhrng[1])+1)
    mhgrid  = isomh[jmhmin:jmhmax]

    # Size of grid in each dimension
    nage    = len(agegrid)
    nmh     = len(mhgrid)
    nmass   = len(massgrid)
    print("nage, nmh, nmass: %d, %d, %d" % (nage, nmh, nmass))

    # Redetermine age and metallicity ranges based on available isochrones
    agerng = (np.min(agegrid), np.max(agegrid))
    mhrng = (np.min(mhgrid), np.max(mhgrid))

    # MultiIndex dataframe for applying transformations
    index = pd.MultiIndex.from_product([agegrid, mhgrid], names = ['age','mh'])
    age_mh = pd.DataFrame(list(product(agegrid, mhgrid)), columns=['age','mh'])

    # Isochrone string identifiers
    #age_mh['isoname'] = "age"+age_mh.age.astype(str)+"mh"+age_mh.mh.astype(str)
    age_mh['isoname'] = age_mh.apply(lambda row: stringLength(row.age, row.mh, iso_info), axis=1)

    # Absolute magnitude arrays from isodict
    age_mh['absA'] = age_mh.isoname.map(lambda x: iso_info[x].Aabs)
    age_mh['absB'] = age_mh.isoname.map(lambda x: iso_info[x].Babs)
    age_mh['absC'] = age_mh.isoname.map(lambda x: iso_info[x].Cabs)

    # Create colour column and extend masses to full length to include zero values for matrix purposes.
    age_mh['ABcol'] = age_mh.absA - age_mh.absB

    # Restack age_mh to create matrices in colour and magnitude
    age_mh.set_index(['age','mh'], inplace=True)
    absAmat = np.array(age_mh[['absA']].unstack()).tolist()
    absAmat = np.array(absAmat)
    absBmat = np.array(age_mh[['absB']].unstack()).tolist()
    absBmat = np.array(absBmat)
    absCmat = np.array(age_mh[['absC']].unstack()).tolist()
    absCmat = np.array(absCmat)
    ABcolmat = np.array(age_mh[['ABcol']].unstack()).tolist()
    ABcolmat = np.array(ABcolmat)

    # Expand Magnitude grids to account for central coordinates
    absAmat, agegridAabs = AM.extendGrid(absAmat, agegrid, axis=0, 
                                        x_lbound=True, x_lb=0.)
    absAmat, mhgridAabs = AM.extendGrid(absAmat, mhgrid, axis=1)
    absAmat, massgridAabs = AM.extendGrid(absAmat, massgrid, axis=2, 
                                        x_lbound=True, x_lb=0., x_ubound=True, x_ub=1.)
    # Expand Magnitude grids to account for central coordinates
    absBmat, agegridBabs = AM.extendGrid(absBmat, agegrid, axis=0, 
                                        x_lbound=True, x_lb=0.)
    absBmat, mhgridBabs = AM.extendGrid(absBmat, mhgrid, axis=1)
    absBmat, massgridBabs = AM.extendGrid(absBmat, massgrid, axis=2, 
                                        x_lbound=True, x_lb=0., x_ubound=True, x_ub=1.)
    # Expand Magnitude grids to account for central coordinates
    absCmat, agegridCabs = AM.extendGrid(absCmat, agegrid, axis=0, 
                                        x_lbound=True, x_lb=0.)
    absCmat, mhgridCabs = AM.extendGrid(absCmat, mhgrid, axis=1)
    absCmat, massgridCabs = AM.extendGrid(absCmat, massgrid, axis=2, 
                                        x_lbound=True, x_lb=0., x_ubound=True, x_ub=1.)
    # Expand Colour grids to account for central coordinates
    ABcolmat, agegridCol = AM.extendGrid(ABcolmat, agegrid, axis=0, 
                                        x_lbound=True, x_lb=0.)
    ABcolmat, mhgridCol = AM.extendGrid(ABcolmat, mhgrid, axis=1)
    ABcolmat, massgridCol = AM.extendGrid(ABcolmat, massgrid, axis=2, 
                                        x_lbound=True, x_lb=0., x_ubound=True, x_ub=1.)

    # Interpolate over matrices to get col&mag as a function of age, metallicity, mass
    print((np.min(agegridCol), np.max(agegridCol)), (np.min(mhgridCol), np.max(mhgridCol)), (np.min(massgridCol), np.max(massgridCol)))
    magA_interp = RGI((agegridAabs, mhgridAabs, massgridAabs), absAmat, bounds_error=False, fill_value=np.nan)
    magB_interp = RGI((agegridBabs, mhgridBabs, massgridBabs), absBmat, bounds_error=False, fill_value=np.nan)
    magC_interp = RGI((agegridCabs, mhgridCabs, massgridCabs), absCmat, bounds_error=False, fill_value=np.nan)
    col_interp = RGI((agegridCol, mhgridCol, massgridCol), ABcolmat, bounds_error=False, fill_value=np.nan)

    colrng = (np.min(ABcolmat), np.max(ABcolmat))
    magrng = (np.min(absCmat), np.max(absCmat))

    return magA_interp, magB_interp, magC_interp, col_interp, agerng, mhrng, magrng, colrng


class NearestIsochrone:

    '''
    NearestIsochrone - Calculate col/mag from nearest isochrone method

    Parameters
    ----------
        isoFile: str
            - Path to the isochrone data file

    Functions
    ---------
        

    Returns
    -------
        appMag(age, mh, mass, s)

    '''
    
    def __init__(self, isoFile):
        
        # Unpickle Isochrone files
        with open(iso_pickle, "rb") as f:
            pi = pickle.load(f)
        # Assign interpolant properties to class attributes
        self.isoage    = np.copy(pi['isoage'])
        self.isomh    = np.copy(pi['isomh'])
        self.isodict   = pi['isodict']
        # Clear pi from memory
        del(pi)
        gc.collect()
        
        # Conversion of absolute to apparent magnitude
        self.appmag = lambda absmag, s: absmag + 5*np.log10(s*1000/10)
        # Conversion of apparent to absolute magnitude
        self.absmag = lambda appmag, s: appmag - 5*np.log10(s*1000/10)
        
    def __call__(self, age, mh, mass, s):

        '''
        __call__ - Returns the apparent magnitude given age, metallicity, mass and distance

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated
            s - array or float (kpc)
                distance of objects to be calculated     

        Returns
        -------
            appMag(age, mh, mass, s)

        '''
        
        return self.appMag(age, mh, mass, s)
        
    def nearestVal(self, l, x):

        '''
        nearestVal - Takes a list and a value and returns the nearest value in the list

        Parameters
        ----------
            l: list or array of float
                - List of values to check through
            x: float
                - Value which we want list element to be closest to.
        Returns
        -------
            l[index0 + plus]: float
                - Value in list which is closest to x
        '''
        
        # Find absolute difference between all elements
        diff = np.abs(x-l)
        # Take the list val which has difference as the minimum
        listval = l[diff == np.min(diff)][0]

        return listval
    
    def absMag(self, age, mh, mass):

        '''
        absMag - Calculate absolute magnitude in J, H, and K from age, metallicity and mass

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated

        Inherited
        ---------
            isoage - 1D array
                Set of age values in isochrones.
            isomh  - 1D array
                Set of metallicity values in isochrones.
            isodict - dict
                Dictionary containing all isochrones.

        Returns
        -------
            J, H, K: 1D array of float
                - J, H and K absolute magnitudes
        '''
        
        # Round age to nearest
        age_rd = self.nearestVal(self.isoage, age)
        # Round mh to nearest
        mh_rd = self.nearestVal(self.isomh, mh)
        
        # Label of isochrone
        isoname = "age"+str(age_rd)+"mh"+str(mh_rd)
        isochrone = self.isodict[isoname]
        
        Mi = isochrone[:,2]
        J = isochrone[:,13]
        H = isochrone[:,14]
        K = isochrone[:,15]
        
        isointerp_MJ = scipy.interpolate.interp1d(Mi, J)#, bounds_error=False, fill_value=np.nan)
        isointerp_MH = scipy.interpolate.interp1d(Mi, H)#, bounds_error=False, fill_value=np.nan)
        isointerp_MK = scipy.interpolate.interp1d(Mi, K)#, bounds_error=False, fill_value=np.nan)
        
        J = isointerp_MJ(mass)
        K = isointerp_MK(mass)
        H = isointerp_MH(mass)
        
        return J, K, H
    
    def appMag(self, age, mh, mass, s):

        '''
        appMag - Calculate apparent magnitude in j, h, k from age, metallicity, mass and distance

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated
            s - array or float (kpc)
                distance of objects to be calculated     

        Inherited
        ---------
            appmag: lambda
                - Convert absolute to apparent magnitude given distance

        Returns
        -------
            j, h, k: 1D array of float
                - j, h, k apparent magnitudes
        '''
        
        J, K, H = self.absMag(age, mh, mass)
        
        [j,k,h] = self.appmag([J,K,H], s)
        
        return j, k, h

def stringLength(age, mh, isodict):

        # Retrieve isochrones from pi
        try:
            interpname  = "age"+str(round(age, 14))+"mh"+"{}".format(mh)
            isochrone   = isodict[interpname]
        except KeyError:
            try:
                interpname  = "age"+str(round(age, 13))+"mh"+"{}".format(mh)
                isochrone   = isodict[interpname]
            except KeyError:
                interpname  = "age"+str(round(age, 12))+"mh"+"{}".format(mh)
                isochrone   = isodict[interpname]

        return interpname

class mScale():

    '''
    ##### NOT IN USE #####
    mScale - Class for converting between mass and scaled mass with max and min mass interps.

    Parameters
    ----------
        Mmax_interp: RGI
            - Interpolant of maximum mass of isochrones over age and metallicity
        Mmin: float
            - Min mass of all isochrones (seems to be the same value for every isochrone
    '''
    
    def __init__(self, Mmax_interp, Mmin):
        
        self.Mmax_interp = Mmax_interp
        self.Mmin = Mmin
        
        self.function = lambda mass, Mmax, Mmin: (mass-Mmin)/(Mmax-Mmin)
        
    def __call__(self, age, mh, mass):
        
        '''
        __call__ - Return mass scaled given age, metallicity and mass

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                ages of objects to be calculated
            mh - array or float (dex)
                metallicity of objects to be calculated
            mass - array or float (solar mass)
                mass of objects to be calculated
        Inherited
        ---------
            function: lambda
                - Calculate scaled mass from mass, Mmax, Mmin

        Returns
        -------
            mass_scaled: arr of float  
                - Scaled mass for each object in the array
        '''

        mass_scaled = self.function( mass, self.Mmax_interp((age, mh)), self.Mmin)
    
        return mass_scaled
    
class mUnscale():

    '''
    ##### NOT IN USE #####
    mScale - Class for converting between scaled mass and mass with max and min mass interps.

    Parameters
    ----------
        Mmax_interp: RGI
            - Interpolant of maximum mass of isochrones over age and metallicity
        Mmin: float
            - Min mass of all isochrones (seems to be the same value for every isochrone
    '''
    
    def __init__(self, Mmax_interp, Mmin):
        
        self.Mmax_interp = Mmax_interp
        self.Mmin = Mmin
        
        self.function = lambda mass_scaled, Mmax, Mmin: mass_scaled*(Mmax-Mmin) + Mmin
        
    def __call__(self, age, mh, mass_scaled):

        '''
        __call__ - Return mass given age, metallicity and scaled mass

        Parameters
        ----------
            age - array or float (same size array as mh, mass, s) (Gyr)
                - ages of objects to be calculated
            mh - array or float (dex)
                - metallicity of objects to be calculated
            mass_scaled - array or float in range [0, 1]
                - scaled mass of objects to be calculated
        Inherited
        ---------
            function: lambda
                - Calculate scaled mass from mass, Mmax, Mmin

        Returns
        -------
            mass - array or float (solar mass)
                - mass of objects to be calculated
        '''
    
        m = self.function( mass_scaled, self.Mmax_interp((age, mh)), self.Mmin)
        
        return m