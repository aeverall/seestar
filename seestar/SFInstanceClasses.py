'''
SFInstanceClasses - Set of classes for defining and calculating selection functions in observable and intrinsic coordinates.

Classes
-------
    observableSF - Selection Function in observable coordinates (magnitude and colour) for a single field.
        - Returns the value of the SF interpolant of the given field at the given coordinates.

Functions
---------
    setattrs - Sets the attributes of a class from a dictionary

    functionIMFKoupra - Calculate initial mass function weight of mass value.

    intrinsicSF - Intrinsic selection function determined with Isochrone Calculator

    intrinsicIMFSF - Intrinsic selection function determined with Isochrone Calculator
        - Integrated over the initial mass function so not dependent on mass

Requirements
------------

'''

import numpy as np
from seestar import StatisticalModels

def obsSF_dicttoclass(obsSF_dicts):

    '''
    obsSF_dicttoclass - Convert dict-of-dicts to dict-of-class instances

    Parameters
    ----------
        obsSF_dicts: dict of dicts
            - Dictionary of dictrionaries of observableSF class instances

    Returns
    -------
        obsSF_classes: dict of class instances
            - Dictionary of instances of observableSF class
    '''

    obsSF_classes = {}
    for field in obsSF_dicts: # Load classes from dictionaries

        # Initialise class instance
        obsSF_field = observableSF(field)
        # Set class attributes from dictionary
        setattrs(obsSF_field, **obsSF_dicts[field])

        # Load models from DF and SF dictionaries
        DF_model = getattr(StatisticalModels, obsSF_field.DF_model['modelname'])
        SF_model = getattr(StatisticalModels, obsSF_field.SF_model['modelname'])
        DF_model = DF_model(runscaling=False)
        SF_model = SF_model(runscaling=False)
        # Set attributes in models
        setattrs(DF_model, **obsSF_field.DF_model)
        setattrs(SF_model, **obsSF_field.SF_model)
        # Reassign model class instance to obsSF class
        obsSF_field.DF_model = DF_model
        obsSF_field.SF_model = SF_model

        # Add class instance to dictionary
        obsSF_classes[field] = obsSF_field

    return obsSF_classes

class observableSF():

    '''
    observableSF - Selection Function in observable coordinates (magnitude and colour) for a single field.
        - Returns the value of the SF interpolant of the given field at the given coordinates.

    Parameters
    ----------
        fieldID: IDtype --- Is this actually needed?
            - The label of the field being iterated

    Functions
    ---------
        __call__ - Calculate the value of the selection function interpolant at the given 
            magnitude and colour (x and y).
        save - Converts attributes of class to a dictionary and saves the dictionary.
        load - Loads attributes of class from the saved dictionary.
    '''

    def __init__(self, fieldID):

        self.field = fieldID

        # mag_range, col_range are now the maximum and minimum values of grid centres used in the RGI.
        self.DF_model = None
        self.DF_magrange = None
        self.DF_colrange = None
        # mag_range, col_range are now the maximum and minimum values of grid centres used in the RGI.
        self.SF_model = None
        self.SF_magrange = None
        self.SF_colrange = None

    def __call__(self, xy):

        '''
        __call__ - Calculate the value of the selection function interpolant at the given 
            magnitude and colour (x and y).

        Parameters
        ----------
            xy: tuple of float or arrays
                - x and y coordinates to be evaluated.

        Inherited
        ---------
            SF_interp: RGI
                - Selection function interpolant for field
            SF_magrange: tuple of float
                - Minimum and maximum magnitudes of observing window.
            SF_colrange: tuple of float
                - Minimum and maximum colours of observing window.

        Returns
        -------
            SF: float or array
                - Selection Function values for x and y coordinates
        '''
        x, y = xy
        SFval = self.SF_model(*(x, y))

        SFval[(x<self.SF_magrange[0])|(x>self.SF_magrange[1])|\
            (y<self.SF_colrange[0])|(y>self.SF_colrange[1])] = 0.

        return SFval

    def save(self, filename):

        '''
        save - Converts attributes of class to a dictionary and saves the dictionary.

        Parameters
        ----------
            filename: str
                - File where data is saved.
        '''

        # Convert attributes to dictionary
        attr_dict = vars(self)

        # Dump pickled dictionary of attributes
        with open(filename, 'wb') as handle:
            pickle.dump(attr_dict, handle)

    def load(self, filename):

        '''
        load - Loads attributes of class from the saved dictionary.

        Parameters
        ----------
            filename: str
                - File where data is saved.
        '''

        # Load pickled dictionary of attributes
        with open(filename, "rb") as input:
            file_dict  = pickle.load(input) 

        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])

class intrinsicSF():

    '''
    intrinsicSF - Intrinsic selection function determined with Isochrone Calculator

    Functions
    ---------
        __call__ - Calculate the intrinsic selection function from the intrinsic coordinates.
        save - Converts attributes of class to a dictionary and saves the dictionary.
        load - Loads attributes of class from the saved dictionary.

    Dependencies
    ------------
        IsoCalculator: IsochroneScaling.IntrinsicToObservable instance
            - Class with functions for converting between intrinsic and observable coordinates
    '''


    def __init__(self):

        # Calculator for getting col, mag from age, mh, s
        self.IsoCalculator = None

    def __call__(self, agemhmasss, obsSF):

        '''
        __call__ - Calculate the intrinsic selection function from the intrinsic coordinates.

        Parameters
        ----------
            agemhmasss: tuple of floats or arrays: (age, mh, mass, s)
                - age, mh, mass and s of stars for which the selection function is being calculated
            obsSF: observableSF instance
                - Class for calculating the observable selection function from colour and magnitude

        Inherited
        ---------
            IsoCalculator: IsochroneScaling.IntrinsicToObservable instance
                - Class with functions for converting between intrinsic and observable coordinates

        Returns
        -------
            SF: float or array
                - Selection function for given intrinsic coordinates.
        '''
        age, mh, mass, s = agemhmasss
        # Calculate colour and absolute magnitude from interpolants
        col, mag = self.IsoCalculator(age, mh, mass, s)
        # Selection function from SF in observed coordinates
        SF = obsSF((mag, col))

        return SF

    def save(self, filename):

        '''
        save - Converts attributes of class to a dictionary and saves the dictionary.

        Parameters
        ----------
            filename: str
                - File where data is saved.
        '''

        # Convert attributes to dictionary
        attr_dict = vars(self)

        # Dump pickled dictionary of attributes
        with open(filename, 'wb') as handle:
            pickle.dump(attr_dict, handle)

    def load(self, filename):

        '''
        load - Loads attributes of class from the saved dictionary.

        Parameters
        ----------
            filename: str
                - File where data is saved.
        '''

        # Load pickled dictionary of attributes
        with open(filename, "rb") as input:
            file_dict  = pickle.load(input) 

        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])

class intrinsicIMFSF():

    '''
    intrinsicIMFSF - Intrinsic selection function determined with Isochrone Calculator
        - Integrated over the initial mass function so not dependent on mass

    Functions
    ---------
        __call__ - Calculate the intrinsic selection function from the intrinsic coordinates.
        save - Converts attributes of class to a dictionary and saves the dictionary.
        load - Loads attributes of class from the saved dictionary.

    Dependencies
    ------------
        IsoCalculator: IsochroneScaling.IntrinsicToObservable instance
            - Class with functions for converting between intrinsic and observable coordinates
    '''

    def __init__(self):

        # Calculator for getting col, mag from age, mh, s
        self.IsoCalculator = None

        # List of masses to integrate over
        self.mass_scaled = np.linspace(0.01, 0.99, 500)

    def __call__(self, agemhs, obsSF):

        '''
        __call__ - Calculate the intrinsic selection function from the intrinsic coordinates.

        Parameters
        ----------
            agemhs: tuple of floats or arrays: (age, mh, s)
                - age, mh, mass and s of stars for which the selection function is being calculated
            obsSF: observableSF instance
                - Class for calculating the observable selection function from colour and magnitude

        Inherited
        ---------
            IsoCalculator: IsochroneScaling.IntrinsicToObservable instance
                - Class with functions for converting between intrinsic and observable coordinates
            mass_scaled: array
                - Set of scaled mass values in range [0, 1] which will be integrated over with the IMF

        Returns
        -------
            SF: float or array
                - Selection function for given intrinsic coordinates.
        '''

        age, mh, s = agemhs

        self.mass_scaled = np.linspace(0.01, 0.99, 500)

        # Convert values into grids expanded over mass values
        agegrid = np.repeat([age,], len(self.mass_scaled), axis=0)
        mhgrid = np.repeat([mh,], len(self.mass_scaled), axis=0)
        sgrid = np.repeat([s,], len(self.mass_scaled), axis=0)

        massgrid = self.mass_scaled
        Ndim = len(np.shape(age))
        for i in range( Ndim ):
            index = Ndim - (i+1)
            massgrid = np.repeat([massgrid,], np.shape(age)[index], axis=0)
        gridDim = len( np.shape(agegrid) )
        axes = np.linspace(0, gridDim-1, gridDim).astype(int)
        axes[0] = axes[len(axes)-1]
        axes[1:] -= 1
        massgrid = np.transpose( massgrid, axes=axes )

        # Unscale the mass and calculate the IMF contribution
        Mmingrid = self.IsoCalculator.Mmin_interp((agegrid, mhgrid))
        Mmaxgrid = self.IsoCalculator.Mmax_interp((agegrid, mhgrid))
        m_scaledgrid = self.IsoCalculator.unscaling( massgrid, Mmaxgrid, Mmingrid )
        weightgrid = functionIMFKoupra( massgrid )

        # Find colour and apparent magnitude values for age, mh, m_scaled, s
        col, mag = self.IsoCalculator(agegrid, mhgrid, m_scaledgrid, sgrid)

        # Integrate over IMF
        SF = np.sum( obsSF((mag, col)) * weightgrid , axis=0 )
        # Normalisation factors for IMF
        Norm = np.sum( weightgrid , axis=0 )
        SF = SF/Norm

        return SF

    def save(self, filename):

        '''
        save - Converts attributes of class to a dictionary and saves the dictionary.

        Parameters
        ----------
            filename: str
                - File where data is saved.
        '''

        # Convert attributes to dictionary
        attr_dict = vars(self)

        # Dump pickled dictionary of attributes
        with open(filename, 'wb') as handle:
            pickle.dump(attr_dict, handle)

    def load(self, filename):

        '''
        load - Loads attributes of class from the saved dictionary.

        Parameters
        ----------
            filename: str
                - File where data is saved.
        '''

        # Load pickled dictionary of attributes
        with open(filename, "rb") as input:
            file_dict  = pickle.load(input) 

        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])

    def attr_dict(self):

        # Dictionary of attributes
        return vars(self)


def setattrs(_self, **kwargs):

    '''
    setattrs - Sets the attributes of a class from a dictionary

    **kwargs
    --------
        _self: class instance
            - Instance of class having attributes set.

        dictionary containing all attributes of the class.
    '''

    for k,v in kwargs.items():
        setattr(_self, k, v)

# Koupra's initial mass function
def functionIMFKoupra(mass):

    '''
    functionIMFKoupra - Calculate initial mass function weight of mass value.

    Parameters
    ----------
        mass: float or array
            - Initial mass of objects for calculating IMF weight.

    Returns
    -------
        IMF: float or array
            - Initial mass function value for given mass.
    '''

    a = 10.44
    IMF = np.zeros(np.shape(mass))

    con1 = (mass<0.08)
    con2 = (mass>0.08)&(mass<0.5)
    con3 = (mass>0.5)

    IMF[con1] = a * (mass[con1]**(-0.3))
    IMF[con2] = a * 0.08 * (mass[con2]**(-1.3))
    IMF[(mass>0.5)] = a * 0.08 * 0.5 * (mass[con3]**(-2.3))

    return IMF
