'''


Parameters
----------


**kwargs
--------


Returns
-------


'''

import numpy as np

class observableSF():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    def __init__(self, fieldID):

        self.field = fieldID

        # mag_range, col_range are now the maximum and minimum values of grid centres used in the RGI.
        self.DF_interp = None
        self.DF_gridarea = None
        self.DF_magrange = None
        self.DF_colrange = None
        self.DF_Nside = None
        # mag_range, col_range are now the maximum and minimum values of grid centres used in the RGI.
        self.SF_interp = None
        self.SF_gridarea = None
        self.SF_magrange = None
        self.SF_colrange = None
        self.SF_Nside = None
        
        self.grid_points = None

    def __call__(self, (x, y)):

        SF = self.SF_interp((x, y))

        return SF

    def normalise(self):
        # Not sure what this is meant to do now that it's no longer calculated from spectro/photo
        return self.SF_gridarea/self.DF_gridarea

    def save(self, filename):

        # Convert attributes to dictionary
        attr_dict = vars(self)

        # Dump pickled dictionary of attributes
        with open(filename, 'wb') as handle:
            pickle.dump(attr_dict, handle)

    def load(self, filename):

        # Load pickled dictionary of attributes
        with open(filename, "rb") as input:
            file_dict  = pickle.load(input) 

        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])

class intrinsicSF():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    def __init__(self):

        self.col_interp = None
        self.mag_interp = None

        # Conversion of absolute to apparent magnitude
        self.appmag = lambda absmag, s: absmag + 5*np.log10(s*1000/10)

        # List of masses to integrate over
        self.mass = np.linspace(0.001, 12., 500)
        self.mass_scaled = np.linspace(0.01, 0.99, 500)

        # Conversion from mass to scaled mass
        self.MassScaling = None
        self.MassUnscaling = None

    def __call__(self, (age, mh, s), obsSF):

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

        # Find colour and absolute magnitude values for age, mh, m_scaled
        col = self.col_interp((agegrid, mhgrid, massgrid))
        Mag = self.mag_interp((agegrid, mhgrid, massgrid))
        # Convert absolute to apparent magnitude
        mag = self.appmag(Mag, sgrid)

        # Unscale the mass and calculate the IMF contribution
        massgrid = self.MassUnscaling( agegrid, mhgrid, massgrid )
        weightgrid = functionIMFKoupra( massgrid )

        # Integrate over IMF
        SF = np.sum( obsSF((mag, col)) * weightgrid , axis=0 )
        # Normalisation factors for IMF
        Norm = np.sum( weightgrid , axis=0 )
        SF = SF/Norm

        return SF

class intrinsicMassSF():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    def __init__(self):

        self.col_interp = None
        self.mag_interp = None

        # Conversion of absolute to apparent magnitude
        self.appmag = lambda absmag, s: absmag + 5*np.log10(s*1000/10)

        # Conversion from mass to scaled mass
        self.MassScaling = None

    def __call__(self, (age, mh, mass, s), obsSF):

        # Convert mass to scaled mass for each isochone(interpolated between)
        m_scaled = self.MassScaling(age, mh, mass)
        # Calculate colour and absolute magnitude from interpolants
        col = self.col_interp((age, mh, m_scaled))
        Mag = self.mag_interp((age, mh, m_scaled))
        # Calculate apparent magnitude from distance conversion
        mag = self.appmag(Mag, s)
        # Selection function from SF in observed coordinates
        SF = obsSF((mag, col))

        return SF

class intSF_isocalc():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    def __init__(self):

        # Calculator for getting col, mag from age, mh, s
        self.IsoCalculator = None

        # Range inside which the interpolants give non-nan values
        self.agerng=None
        self.mhrng=None

        # List of masses to integrate over
        self.mass_scaled = np.linspace(0.01, 0.99, 500)

    def __call__(self, (age, mh, s), obsSF):

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

        # Convert attributes to dictionary
        attr_dict = vars(self)

        # Dump pickled dictionary of attributes
        with open(filename, 'wb') as handle:
            pickle.dump(attr_dict, handle)

    def load(self, filename):

        # Load pickled dictionary of attributes
        with open(filename, "rb") as input:
            file_dict  = pickle.load(input) 

        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])

    def attr_dict(self):

        # Dictionary of attributes
        return vars(self)

class intMassSF_isocalc():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    def __init__(self):

        # Calculator for getting col, mag from age, mh, s
        self.IsoCalculator = None

        # Range inside which the interpolants give non-nan values
        self.agerng=None
        self.mhrng=None

    def __call__(self, (age, mh, mass, s), obsSF):

        # Calculate colour and absolute magnitude from interpolants
        col, mag = self.IsoCalculator(age, mh, mass, s)
        # Selection function from SF in observed coordinates
        SF = obsSF((mag, col))

        return SF

    def save(self, filename):

        # Convert attributes to dictionary
        attr_dict = vars(self)

        # Dump pickled dictionary of attributes
        with open(filename, 'wb') as handle:
            pickle.dump(attr_dict, handle)

    def load(self, filename):

        # Load pickled dictionary of attributes
        with open(filename, "rb") as input:
            file_dict  = pickle.load(input) 

        # Convert dictionary to attributes  
        for key in file_dict:
            setattr(self, key, file_dict[key])


def setattrs(_self, **kwargs):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    for k,v in kwargs.items():
        setattr(_self, k, v)

# Koupra's initial mass function
def functionIMFKoupra(mass):

    a = 10.44
    IMF = np.zeros(np.shape(mass))

    con1 = (mass<0.08)
    con2 = (mass>0.08)&(mass<0.5)
    con3 = (mass>0.5)

    IMF[con1] = a * (mass[con1]**(-0.3))
    IMF[con2] = a * 0.08 * (mass[con2]**(-1.3))
    IMF[(mass>0.5)] = a * 0.08 * 0.5 * (mass[con3]**(-2.3))

    return IMF
