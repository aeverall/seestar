'''
DataImport - Contains classes and functions used to import data from databases
             Currently only set up for RAVE database but may be generalised to other databases

Classes
-------
     DataImport - Class for importing data from downloaded RAVE databases

Functions
---------
    ApparentToAbolute - Transformation from apparent magnitude to absolute magnitude

    AngleAnalysis - Imports data and creates dataframe with information required to analyse Equatorial / Galactic angles of data

    IsochroneAnalysis - Imports data and creates dataframe with information required to analyse Isochrones against McMillan data

    PjmDataAnalysis - Imports data and creates dataframe with information required to analyse reliability/accuracy of McMillan spectroscopic data

    PjmErrorAnalysis - Imports dataframe with relevant information for analysing multi-Gaussian distributions of probability density functions

    SelectionGrid - Imports dataframe with relevant information for creating grids of data to develope the selection function

    RavePlates - Imports dataframe with relevant information for analysing positions/distributions of RAVE plates    

    FbfData - Imports dataframe with Wojno's positions of RAVE plates in Equatorial angles (RA,Dec)

    PbpData - Imports dataframe with Wojno's equal-area sky pixelation

    ListIt - Prints a list of all RAVE_DR5 fields

    TmassToFile - Converts 2MASS datafile into the correct format with column headers

Requirements
------------
agama

Access to path containing Galaxy modification programes:
    sys.path.append("../FitGalMods/")

    from MilkyWayDFModified import MilkyWayDF as mwdf
    from GalDFModified import GalDF
    import CoordTrans

ArrayMechanics
AngleDisks
PlotCoordinates
DataImport
'''

import pandas as pd
import numpy as np

from MilkyWayDFModified import MilkyWayDF as mwdf
from GalDFModified import GalDF
import CoordTrans

import gc

from os.path import join
root = "../../Project/Data"

from AngleDisks import *

class DataImport():
    
    '''
    DataImport - Class for importing data from downloaded RAVE databases

    Parameters
    ----------
        data_set: string
                Identifier for the database which the data will be taken from

    Functions
    ---------
        ImportData - Function which actively brings data in from csv file to a pandas dataframe

        ListColumns - Returns a list of all column headers which are held within the database

        CorrectUnits - Transforms the input units to output units. 
                       Units are described within the dictionary: self.field_description
        

    Returns
    -------
        self.df: Pandas dataframe
                DataFrame of database which has been transformed and adjusted to contain required fields 
    '''

    def __init__(self, data_set):
        
        self.location = {'PJM': "../../Project/Data/Distances_PJM2017.csv",
                         'RAVE_TGAS': "../../Project/Data/RAVE_TGAS.csv",
                         'RAVE_DR5': "../../Project/Data/RAVE_DR5.csv"}
        
        
        RAVE_TGAS_fields = ['RAVE_OBS_ID', 'MatchFlag_TGAS', 'tycho2_id',
                            'ra', 'ra_error','dec','dec_error','parallax','parallax_error',
                             'pmra','pmra_error','pmdec','pmdec_error', 
                             'l','b','ecl_lon','ecl_lat']
        RAVE_DR5_fields = ['RAVE_OBS_ID', 'MatchFlag_TGAS', 'ID_TYCHO2',
                           'RAdeg', 'DEdeg',
                           'RA_TGAS', 'DE_TGAS', 'parallax_TGAS','parallax_error_TGAS',
                           'pmRA_TGAS', 'pmRA_error_TGAS','pmDE_TGAS','pmDE_error_TGAS', 'HRV', 'eHRV',
                           'logg_K', 'Met_K', 'Teff_K', 'Teff_IR']
        PJM_fields = ['RAVE_OBS_ID', 'distance', 'age', 
                      'parallax', 'eparallax',
                      'parallax_TGAS', 'parallax_error_TGAS',
                      'logg_PJM', 'Teff_PJM']
        magnitudes = ['Jmag_2MASS', 'Kmag_2MASS', 'Hmag_2MASS']

        self.fields = {'PJM': PJM_fields,
                       'RAVE_TGAS': RAVE_TGAS_fields,
                       'RAVE_DR5': RAVE_DR5_fields,
                       'magnitudes': magnitudes}

        self.data_set = data_set
        
        self.df = pd.DataFrame()

        self.solpos = np.array([8.3,0.014,-14.0,12.24+238.,7.25])
        
        #Flags for determining progress in data mining
        self.Imported_bool = False
        self.CorrectUnits_bool = False
        self.Galactic_bool = False
        self.Cartesian_bool = False
        self.aUVW_bool = False
        self.AngActFre_bool = False
        
        #Coordinate labels for each coordinate system
        self.Equatorial_coords = ['RA_TGAS', 'DE_TGAS', 's', 'HRV', 'pmRA_TGAS', 'pmDE_TGAS']
        self.Galactic_coords = ['l', 'b', 's_gal', 'vlos', 'mulcosb', 'mub']
        self.aUVW_coords = ['alpha', 'U', 'V', 'W']
        self.Act_Ang_Freq_coords = ['Jr', 'Jz', 'Jphi',
                               'THr', 'THz', 'THphi',
                               'Or', 'Oz', 'Ophi']

        self.ImportMagnitudes = False

        self.field_description = {'RAVE_DR5':
                                        {'RAVE_OBS_ID': '',
                                        'MatchFlag_TGAS': '',
                                        'ID_TYCHO2': '',
                                        'RAdeg': 'degrees',
                                        'DEdeg': 'degrees',
                                        'RA_TGAS': '', 
                                        'DE_TGAS':  '',
                                        'parallax_TGAS': '',
                                        'parallax_error_TGAS': '',
                                        'pmRA_TGAS': '', 
                                        'pmRA_error_TGAS': '',
                                        'pmDE_TGAS': '',
                                        'pmDE_error_TGAS': '', 
                                        'HRV': '', 
                                        'eHRV': '',
                                        'logg_K': '', 
                                        'Met_K': '', 
                                        'Teff_K': '', 
                                        'Teff_IR': ''},
                                    'PJM':
                                        {'RAVE_OBS_ID': '',
                                        'distance': '',
                                        'age': '',
                                        'parallax': '',
                                        'eparallax': '',
                                        'parallax_TGAS': '', 
                                        'parallax_error_TGAS': '',
                                        'logg_PJM': '',
                                        'Teff_PJM': ''}
                                    }
        
    def ImportData(self, **kwargs):

        '''
        ImportData - Function which actively brings data in from csv file to a pandas dataframe

        **kwargs
        --------
            fields: List of strings - self.fields[self.data_set]
                    - List of column headers to be included in the dataframe
                    - If not provided, automatic list from self.fields is used

        Returns
        -------
            None
            Prints out 'Data Imported when complete'

        '''

        if self.ImportMagnitudes: #Include JHK magnitudes
            self.fields[self.data_set].extend(self.fields['magnitudes'])

        # Fields may be replaced by set of fields given in the description
        options = {'fields': self.fields[self.data_set]}
        options.update(kwargs)
        self.fields[self.data_set] = options['fields']
        
        fields = options['fields']
        filen = self.location[self.data_set]
        df = pd.read_csv(filen, usecols = fields)
        df.set_index("RAVE_OBS_ID", inplace = True)
        self.df = df

        del(df)
        gc.collect()

        self.Imported_bool = True
        print('Data Imported')
    
    def ListColumns(self):

        '''
        ListColumns - Returns a list of all column headers which are held within the database

        Inherited
        ---------
            self.location: dictionary of strings
                    - matches data_set labels to paths to the csv files containing data

            self.data_set: string 
                    - labels for individual databases from RAVE

        Returns
        -------
            columnn_headers: list of strings
                    - list of all columns which are contained within the database (data_set)
        '''
        
        filen = self.location[self.data_set]
        df = pd.read_csv(filen, nrows = 0)
        
        column_headers = list(df)
        
        return column_headers
    
    def CorrectUnits(self, **kwargs):

        '''
        CorrectUnits - Transforms the input units to output units. 
                       Units are described within the dictionary: self.field_description

        Inherited
        ---------
            self.Imported_bool: bool
                    - Changed to true once the ImportData function has been run

            self.df: dataframe
                    - Dataframe in which the imported and adjusted data is stored

            self.CorrectUnits_bool: bool
                    - Changed to true once the CorrectUnits function has been run

        **kwargs
        --------
            distance: bool - True
                    - If true then a new column 's' is created from 1/parallax_TGAS

        Returns
        -------
            None
            prints 'Units Corrected' when complete
        '''

        options = {'distance': True}
        options.update(kwargs)
        
        if self.Imported_bool: pass
        else: self.ImportData()
        
        df = self.df
        
        #Convert parallax to distance
        if options['distance']: df['s'] = 1/df.parallax_TGAS

        #Convert degrees to radians 
        df.RA_TGAS *= 2*np.pi/360
        df.DE_TGAS *= 2*np.pi/360
        
        self.df = df

        del(df)
        gc.collect()
        
        #set flag to true
        self.CorrectUnits_bool = True
        print('Units Corrected')
        
    def AddGalactic(self):

        '''
`       AddGalactic - Introduces new columns for df for GalacticCoordinates

        Inherited
        ---------
            self.CorrectUnits_bool: bool
                    - Changed to true once the CorrectUnits function has been run
                    - If False, automatically runs self.CorrectUnits()

            self.Equatorial_coords: list of strings
                    - List of column headers which correspond to 6D Equatorial coordinates

            self.Galactic_coords: list of strings
                    - List of column headers which correspond to 6D Galactic coordinates

            self.df: dataframe
                    - Dataframe in which the imported and adjusted data is stored

            self.Galactic_bool: bool
                    - Changed to true once the AddGalactic function has been run

        Returns
        -------
            None
            prints 'Galactic Coords Created' when complete


        '''
        
        if self.CorrectUnits_bool: print('Units are correct, creating Galactic coordinates')
        else: 
            print('Units are not corrected, correcting units before creating Galactic coordinates')
            self.CorrectUnits()
        
        df = self.df
        
        Equatorial_coords = self.Equatorial_coords
        Galactic_coords = self.Galactic_coords
        
        # Transform equatorial to galactic
        Equatorial = np.array((df[Equatorial_coords]))
        Galactic = CoordTrans.EquatorialToGalactic(Equatorial)
        
        # Add new coordinates to the df DataFrame
        Galactic = pd.DataFrame(Galactic, columns = Galactic_coords)
        Galactic.index = df.index
        df = df.join(Galactic)
        
        self.df = df
        del(df)
        gc.collect()

        self.Galactic_bool = True
        print('Galactic Coords Created')
        
    def AddaUVW(self):

        '''
`       AddUVW - Introduces new columns for df for Cartesian UVW coordinates

        Inherited
        ---------
            self.Galactic_bool: bool
                    - Changed to true once the AddGalactic function has been run
                    - If False, self.AddGalactic() function is run

            self.solpos: list of floats
                    - solar position specified within the MilkyWay relative to the GC

            self.Galactic_coords: list of strings
                    - List of column headers which correspond to 6D Galactic coordinates

            self.aUVW_coords: list of strings
                    - List of column headers which correspond to 4 coordinates: alpha, U, V, W

            self.df: dataframe
                    - Dataframe in which the imported and adjusted data is stored

            self.aUVW_bool: bool
                    - Changed to true once the AddaUVW function has been run

        Returns
        -------
            None
            prints 'aUVW coords created' when complete

        '''
        
        if self.Galactic_bool: print('Galactic coordinates are present, creating aUVW coordinates')
        else: 
            print('Galactic coordinates are no present, creating Galactic coordinates before creating aUVW coordinates')
            self.AddGalactic()
        
        df = self.df    
        solpos = self.solpos
    
        Galactic_coords = self.Galactic_coords
        aUVW_coords = self.aUVW_coords
        
        # Transform Galactic to cartesian
        Galactic = np.array((df[Galactic_coords]))
        aUVW = CoordTrans.CalcaUVW(Galactic, solpos[0])
        # Add new coordinates to df DataFrame
        aUVW = pd.DataFrame(np.column_stack(aUVW), columns = aUVW_coords)
        aUVW.index = df.index
        df = df.join(aUVW)
        
        self.df = df
        del(df)
        gc.collect()

        self.aUVW_bool = True
        print('aUVW coords created')
        
    def AddCartesian(self):
        pass
        
    def AddAngActFre(self):

        '''
`       AddAngActFre - Introduces new columns for df for Angle, Action and Frequency coordinates

        Inherited
        ---------
            self.CorrectUnits_bool: bool
                    - Changed to true once the CorrectUnits function has been run
                    - If False, automatically runs self.CorrectUnits()

            self.solpos: list of floats
                    - solar position specified within the MilkyWay relative to the GC

            self.Equatorial_coords: list of strings
                    - List of column headers which correspond to 6D Equatorial coordinates

            self.Ang_Act_Fre_coords: list of strings
                    - List of column headers which correspond to 9 angle action and frequency coordinates

            self.df: dataframe
                    - Dataframe in which the imported and adjusted data is stored

            self.AngActFre_bool: bool
                    - Changed to true once the AddAngActFre() function has been run

        Returns
        -------
            None
            prints 'Galactic Coords Created' when complete


        '''
        
        if self.CorrectUnits_bool: print('Units are correct, creating Galactic coordinates')
        else: 
            print('Units are not corrected, correcting units before creating Galactic coordinates')
            self.CorrectUnits()
            
        df = self.df  
        solpos = self.solpos

        Equatorial_coords = self.Equatorial_coords
        Act_Ang_Freq_coords = self.Act_Ang_Freq_coords
    
        #Transform into Angle-Action-Frequency coordinates
        solpos = np.array([8.3,0.014,-14.0,12.24+238.,7.25])
        MWDF = mwdf(solpos)
        Equatorial = np.array((df[Equatorial_coords]))
        Actions, Angles, Frequencies = MWDF.ang_act_coords(Equatorial)

        #Correct units:
        Actions = Actions/977.8
        Angles[(Angles[:,2]>np.pi), 2] = Angles[(Angles[:,2]>np.pi), 2] - 2*np.pi
        Angles[(Angles[:,0]>np.pi), 0] = Angles[(Angles[:,0]>np.pi), 0] - 2*np.pi

        # Add new coordinates to df DataFrame
        Act_Ang_Freq = pd.DataFrame(np.column_stack((Actions, Angles, Frequencies)), columns = Act_Ang_Freq_coords)
        Act_Ang_Freq.index = df.index
        df = df.join(Act_Ang_Freq)
        
        self.df = df
        del(df)
        gc.collect()

        self.AngActFre_bool = True        
        print('Action, Angle and Frequency coords created')

    def AddAbsoluteMagnitudes(self, distance_label):

        '''
        AddAbsoluteMagnitudes - Introduces new columns for df for J, K and H absolute magnitudes from 2MASS

        Parameters
        ----------
            distance_label: string
                        - Label of column in df which is used as the distance to calculate the absolute from apparent magnitudes

        Inherited
        ---------
            self.df: dataframe
                    - Dataframe in which the imported and adjusted data is stored        

        Returns
        -------
            None
            prints distance_label + ' used for Absolute magnitude calculation' to show which distance measurement is used


        '''

        print(distance_label + ' used for Absolute magnitude calculation')

        df = self.df

        df['Jmag_abs'] = ApparentToAbolute(df.Jmag_2MASS, getattr(df, distance_label))
        df['Kmag_abs'] = ApparentToAbolute(df.Kmag_2MASS, getattr(df, distance_label))
        df['Hmag_abs'] = ApparentToAbolute(df.Hmag_2MASS, getattr(df, distance_label))

        self.df = df
        del(df)
        gc.collect()


def ApparentToAbolute(Apparent, p):

    '''
    ApparentToAbolute - Transformation from apparent magnitude to absolute magnitude

    Parameters
    ----------
        Apparent: Series/Array
                - Apparent magnitudes of stars

        p: Series/Array
                - Parallaxes of stars

    Returns
    -------
        Absolute: Series/Array
                - Absolute magnitudes of stars

    '''

    #Convert p in mas to s in pc
    s = 1000/(p)
    Absolute = Apparent - 2.5*np.log10((s/10)**2)
    return Absolute

def AngleAnalysis():

    '''
    AngleAnalysis - Imports data and creates dataframe with information required to analyse Equatorial / Galactic angles of data

    Returns
    -------
        importer.df: DataFrame
                - Data imported having had units corrected and filtered for relevant stars

    Uses class DataImport
    '''

    importer = DataImport('RAVE_DR5')
    importer.ImportData()
    importer.CorrectUnits()
    importer.AddGalactic()

    importer.df = importer.df[pd.notnull(importer.df.l)]

    return importer.df

def IsochroneAnalysis():

    '''
    IsochroneAnalysis - Imports data and creates dataframe with information required to analyse Isochrones against McMillan data

    Returns
    -------
        importer.df: DataFrame
                - Data imported having had units corrected and filtered for relevant stars

    Uses class DataImport
    '''

    import_RaveDR5 = DataImport('RAVE_DR5')
    import_PJM = DataImport('PJM')

    RAVE_DR5_fields = ['RAVE_OBS_ID',
                       'parallax_TGAS','parallax_error_TGAS',
                       'Teff_K', 'Teff_IR',
                       'Jmag_2MASS', 'Kmag_2MASS', 'Hmag_2MASS',
                       'logg_K']
    PJM_fields = ['RAVE_OBS_ID', 
                  'age', 
                  'parallax', 'eparallax',
                  'Teff_PJM',
                  'logg_PJM',
                  'mass',
                  'Met_N_K']

    import_RaveDR5.ImportData(fields = RAVE_DR5_fields)
    import_PJM.ImportData(fields = PJM_fields)
    import_RaveDR5.AddAbsoluteMagnitudes('parallax_TGAS')

    df = pd.merge(import_PJM.df, import_RaveDR5.df, how = 'inner', left_index = True, right_index = True)

    return df

def PjmDataAnalysis():

    '''
    PjmDataAnalysis - Imports data and creates dataframe with information required to analyse reliability/accuracy of McMillan spectroscopic data

    Returns
    -------
        importer.df: DataFrame
                - Data imported having had units corrected and filtered for relevant stars

    Uses class DataImport
    '''

    import_PJM = DataImport('PJM')

    PJM_fields = ['RAVE_OBS_ID',
                  'Glon', 'Glat',
                  'age', 'eage', 
                  'parallax', 'eparallax', 
                  'distance', 'edistance',
                  'Teff_PJM', 'eTeff_PJM', 
                  'logg_PJM', 'elogg_PJM', 
                  'logg_N_K', 'elogg_K', 
                  'Teff_IR', 'eTeff_IR', 
                  'Met_N_K', 'eMet_K', 
                  'parallax_TGAS', 'parallax_error_TGAS']

    import_PJM.ImportData(fields=PJM_fields)

    return import_PJM.df

def PjmErrorAnalysis():

    '''
    PjmErrorAnalysis - Imports dataframe with relevant information for analysing multi-Gaussian distributions of probability density functions

    Returns
    -------
        importer.df: DataFrame
                - Data imported having had units corrected and filtered for relevant stars

    Uses class DataImport
    '''

    import_PJM = DataImport('PJM')

    PJM_fields = ['RAVE_OBS_ID', 'RAVEID', 'RAdeg', 'DEdeg', 'Glon', 'Glat', 'HRV', 'eHRV', 'pmRA_TGAS', 'pmRA_error_TGAS', 'pmDE_TGAS', 'pmDE_error_TGAS', 
                  'distance', 'edistance', 'age', 'eage', 'mass', 'e_mass', 'log_Av', 'elog_Av', 'parallax', 'eparallax', 'dist_mod', 'edist_mod', 'Teff_PJM', 'eTeff_PJM', 
                  'logg_PJM', 'elogg_PJM', 
                  'N_Gauss_fit', 
                  'Gauss_mean_1', 'Gauss_sigma_1', 'Gauss_frac_1', 
                  'Gauss_mean_2', 'Gauss_sigma_2', 'Gauss_frac_2', 
                  'Gauss_mean_3', 'Gauss_sigma_3', 'Gauss_frac_3', 
                  'FitQuality_Gauss', 'Fit_Flag_Gauss', 
                  'AV_Schlegel', 'logg_N_K', 'elogg_K', 'Teff_IR', 'eTeff_IR', 'Met_N_K', 'eMet_K', 'parallax_TGAS', 'parallax_error_TGAS', 'Jmag_2MASS', 'eJmag_2MASS', 'Hmag_2MASS', 'eHmag_2MASS', 'Kmag_2MASS', 'eKmag_2MASS', 'W1mag_ALLWISE', 'eW1mag_ALLWISE', 'W2mag_ALLWISE', 'eW2mag_ALLWISE', 'Mg', 'Mg_N', 'Al', 'Al_N', 'Si', 'Si_N', 'Ti', 'Ti_N', 'Fe', 'Fe_N', 'Ni', 'Ni_N', 
                  'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 
                  'SNR', 'Algo_Conv_K', 'flag_lowlogg', 'flag_outlier', 'flag_N', 'flag_pole', 'flag_dup', 'flag_any']
    PJM_fields = ['RAVE_OBS_ID', 'RAVEID', 'RAdeg', 'DEdeg', 'Glon', 'Glat', 'HRV', 'eHRV', 'pmRA_TGAS', 'pmRA_error_TGAS', 'pmDE_TGAS', 'pmDE_error_TGAS', 
                  'distance', 'edistance', 'age', 'eage', 'mass', 'e_mass', 'log_Av', 'elog_Av', 'parallax', 'eparallax', 'dist_mod', 'edist_mod', 'Teff_PJM', 'eTeff_PJM', 
                  'logg_PJM', 'elogg_PJM', 
                  'N_Gauss_fit', 
                  'Gauss_mean_1', 'Gauss_sigma_1', 'Gauss_frac_1', 
                  'Gauss_mean_2', 'Gauss_sigma_2', 'Gauss_frac_2', 
                  'Gauss_mean_3', 'Gauss_sigma_3', 'Gauss_frac_3', 
                  'FitQuality_Gauss', 'Fit_Flag_Gauss', 
                  'AV_Schlegel', 'logg_N_K', 'elogg_K', 'Teff_IR', 'eTeff_IR', 'Met_N_K', 'eMet_K', 'parallax_TGAS', 'parallax_error_TGAS', 
                  'Jmag_2MASS', 'eJmag_2MASS', 'Hmag_2MASS', 'eHmag_2MASS', 'Kmag_2MASS', 'eKmag_2MASS']

    import_PJM.ImportData(fields=PJM_fields)

    df = import_PJM.df
    df['RAVE_OBS_ID'] = df.index
    str_search = '[0-9]*_(?P<plate>[0-9]*(?:m|p)[0-9a-z]*)_[0-9]*'
    ID_extraction = df.RAVE_OBS_ID.astype(str).str.extract(str_search)
    df['plate'] = ID_extraction
     
    return df

def SelectionGrid():

    '''
    SelectionGrid - Imports dataframe with relevant information for creating grids of data to develope the selection function

    Returns
    -------
        importer.df: DataFrame
                - Data imported having had units corrected and filtered for relevant stars

    Uses class DataImport
    '''

    import_RaveDR5 = DataImport('RAVE_DR5')
    import_PJM = DataImport('PJM')

    RAVE_DR5_fields = ['RAVE_OBS_ID',
                        'RA_TGAS', 'DE_TGAS', 
                        'parallax_TGAS', 'parallax_error_TGAS',
                        'Jmag_2MASS', 'eJmag_2MASS', 'Hmag_2MASS', 'eHmag_2MASS', 'Kmag_2MASS', 'eKmag_2MASS']
                        #'FieldName', 'FiberNumber', 'PlateNumber']
    PJM_fields = ['RAVE_OBS_ID',
                  'parallax', 'eparallax']

    import_RaveDR5.ImportData(fields = RAVE_DR5_fields)
    import_RaveDR5.CorrectUnits(distance = False)

    import_PJM.ImportData(fields = PJM_fields)

    df = pd.merge(import_PJM.df, import_RaveDR5.df, how = 'inner', left_index = True, right_index = True)

    return df

def RavePlates():

    '''
    RavePlates - Imports dataframe with relevant information for analysing positions/distributions of RAVE plates

    Returns
    -------
        importer.df: DataFrame
                - Data imported having had units corrected and filtered for relevant stars

    Uses class DataImport
    '''

    import_RaveDR5 = DataImport('RAVE_DR5')

    RAVE_DR5_fields = ['RAVE_OBS_ID',
                        'RA_TGAS', 'DE_TGAS', 
                        'parallax_TGAS', 'parallax_error_TGAS',
                        'pmRA_TGAS', 'pmDE_TGAS', 'HRV',
                        'Jmag_2MASS', 'eJmag_2MASS', 'Hmag_2MASS', 'eHmag_2MASS', 'Kmag_2MASS', 'eKmag_2MASS',
                        'FieldName', 'FiberNumber', 'PlateNumber']

    import_RaveDR5.ImportData(fields = RAVE_DR5_fields)
    import_RaveDR5.CorrectUnits()
    import_RaveDR5.AddGalactic()

    df = import_RaveDR5.df
    df = df[pd.notnull(df.RA_TGAS)]

    df['RAVE_OBS_ID'] = df.index
    str_search = '[0-9]*_(?P<plate>[0-9]*(?:m|p)[0-9a-z]*)_[0-9]*'
    ID_extraction = df.RAVE_OBS_ID.astype(str).str.extract(str_search)
    df['plate'] = ID_extraction

    return df

def FbfData():

    '''
    FbfData - Imports dataframe with Wojno's positions of RAVE plates in Equatorial angles (RA,Dec)

    Returns
    -------
        df: DataFrame
                - Data imported with fields corrected to contain useful/relevant information
    '''

    FBF_fields = ['RAVE_FIELD', 'RAdeg', 'DEdeg', 'OBSDATE', 'N_pointed', 'N_processed', 'N_2MASS']

    df = pd.read_csv(join(root, "RAVE_Completeness_FBF.csv"), usecols = FBF_fields)

    str_search = '[0-9]*_(?P<plate>[0-9]*(?:m|p)[0-9a-z]*)'
    ID_extraction = df.RAVE_FIELD.astype(str).str.extract(str_search)
    df['Plate_ID'] = ID_extraction
    df = df[['RAVE_FIELD', 'Plate_ID', 'RAdeg', 'DEdeg']].groupby('Plate_ID').first()

    return df 

def PbpData():

    '''
    PbpData - Imports dataframe with Wojno's equal-area sky pixelation

    Returns
    -------
        df: DataFrame
                - Data imported with fields corrected to contain useful/relevant information
    '''

    PBP_fields = ['HEALPix32', 
                        'CF00.0', 'CF00.1', 'CF00.2', '.....',
                        'CF13.7', 'CF13.8', 'CF13.9', 'CF14.0']

    df = pd.read_csv(join(root, "RAVE_Completeness_PBP.csv"), usecols = PBP_fields)

    return df 

def ListIt():

    '''
    ListIt - Prints a list of all RAVE_DR5 fields
    '''

    import_RaveDR5 = DataImport('RAVE_DR5')
    listing = import_RaveDR5.ListFields()
    print(listing)

def TmassToFile(input_file, output_file):

        '''
        TmassToFile - Converts 2MASS datafile into the correct format with column headers
                    - Length of a 2MASS file ~ 5m

        Returns
        -------
            None

            Produces a file in 2MASS/Corrected with the rearranged data stored within
        '''
    
        Tmass_path = '../Data/2MASS/'
        
        Headers = pd.read_csv(os.join(Tmass_path, input_file))
        Headers.to_csv(os.join(Tmass_path, output_file), index=False)
        
        # Number of rows compiled at a time
        N=100000
        X=N
        i=0
    
        while X>0:
            print(i)
            data_sample = pd.read_csv(os.join(Tmass_path, input_file), sep='|', 
                                      header = None, nrows = N, names=Headers,
                                      skiprows=i*N)
            data_sample.to_csv(os.join(Tmass_path, output_file),
                               header=False, mode='a+', index=False)
            X = len(data_sample)
            i+=1

def IterateTmass(self, Tmass_path, file_paths):
    
    solidangle = 28.3

    for path in file_paths:
        
        N = 10000
        i = 0
        X = N
        
        while X>0:
            clear = True if i==0 else False
            
            data_sample = pd.read_csv(os.path.join(Tmass_path, path), 
                                      nrows = N,
                                      skiprows=range(1, i*N))
            self.TmassPlateFiles(data_sample, Tmass_path, solidangle, clear=clear)
            
            X = len(data_sample)
            
            i+=1
            if i%10==0: print(i)
            if i==50: break
    
    
    
def TmassPlateFiles(self, stars,
                    path,
                    solidangle,
                    clear = False):

    # Relabel columns
    Tmass_coords = ['ra', 'dec', 'j_m', 'k_m', 'h_m']
    coords = ['RA', 'Dec', 'Jmag', 'Kmag', 'Hmag']

    stars = stars[Tmass_coords]
    stars['SolidAngle'] = np.zeros((len(stars))) + solidangle
    stars = stars.rename(index=str, columns=dict(zip(Tmass_coords, coords)))

    stars = self.PointsToPointings(stars)
    plate_ids = np.unique(stars.loc[:,'point'])
    
    if clear:
        for plate in self.pointings.Plate_ID:
            headers = stars[:0]
            headers.to_csv(os.path.join(path, '2MassPlate_'+plate), 
                           index=False, mode = 'w')
            
    for plate in plate_ids:
        stars.loc[stars.point==plate].to_csv(os.path.join(path, '2MassPlate_'+plate),
                                             index=False,
                                             mode = 'a+', header=False)

def RaveToFile():

    fields = ['RAVE_OBS_ID',
               'RA_TGAS', 'DE_TGAS',
               'Jmag_2MASS', 'Kmag_2MASS', 'Hmag_2MASS', 
               'FieldName']

    df = pd.read_csv('../../Project/Data/RAVE_DR5.csv', usecols=fields)

    strsearch = re.compiled('([0-9]*[a-z][0-9]*)[a-z]*')
    df.FieldName = df.FieldName.astype(str).str.extract(strsearch)

    df.to_csv('../Data/RAVE_wPlateIDs.csv', index=False)

def PointingsToFile():

    df = pd.read_csv('../Data/RAVE_FIELD_Centers.csv', skiprows=1)

    strsearch = re.compile('([0-9]*[a-z][0-9]*)[a-z]*')
    df.RAVE_FIELD = df.RAVE_FIELD.astype(str).str.extract(strsearch)

    df.to_csv('../Data/RAVE_FIELDS_wFieldIDs.csv', index=False)

def reformatTmassPlateFiles(fieldfile, 
                            plate_id,
                            tmass_plate_location):
        
        dffield= pd.read_csv(fieldfile)
    
        listfield = getattr(dffield, plate_id).values.tolist()
        
        database_coords = ['RA','DEC','J','K','H']
        df_coords = ['RA','Dec','Jmag','Kmag','Hmag']
        
        def old_format(field):
                return '00000'+str(int(field))+'.sav'
            
        def new_format(field):
                return str(field)+'.csv'
        
        print(len(listfield))
            
        
        for field in listfield:
            
            try:
                #Correct 2Mass table format
                points = pd.DataFrame(readsav(tmass_plate_location+old_format(field))['fata'])
                headers=list(points)
                points = np.copy(points)
                points = pd.DataFrame(points, columns=headers)[['RA','DEC','J','K','H']]
                points = points.rename(index=str, columns=dict(zip(database_coords, df_coords)))

                points.to_csv(tmass_plate_location+new_format(field))
            except IOError: 
                # due to there not being a sav file for the given field
                pass


def resampleIsochroneMass():
    
    print("Undilling isochrones and interpolants...")
    with open('../Data/stellarprop_parsecdefault_currentmass.dill', "rb") as input:
        pi = dill.load(input)
    print("...done.")
    print(" ")

    isoage = pi.isoage
    isomh = pi.isomh
    isodict = pi.isodict
    
    iso_info = {}
    iso_info['isoage'] = isoage
    iso_info['isomh'] = isomh
    iso_info['isodict'] = {}

    # Koupra's initial mass function
    def functionIMFKoupra(mass):
        a = 10.44
        IMF = np.zeros(len(mass))

        con1 = (mass<0.08)
        con2 = (mass>0.08)&(mass>0.5)
        con3 = (mass>0.5)

        IMF[con1] = a * (mass[con1]**(-0.3))
        IMF[con2] = a * 0.08 * (mass[con2]**(-1.3))
        IMF[(mass>0.5)] = a * 0.08 * 0.5 * (mass[con3]**(-2.3))

        return IMF


    isodict = {}
    i=0

    for age in isoage:
        for mh in isomh:
            interpname  = "age"+str(age)+"mh"+str(mh) 
            isochrone   = pi.isodict[interpname]

            Mi = isochrone[:,3]
            J = isochrone[:,13]
            H = isochrone[:,14]
            K = isochrone[:,15]
            
            isointerp_MiJ = scipy.interpolate.interp1d(Mi, J)
            isointerp_MiH = scipy.interpolate.interp1d(Mi, H)
            isointerp_MiK = scipy.interpolate.interp1d(Mi, K)
            
            Mi_vals = np.linspace(np.min(Mi), np.max(Mi), 100)
            H_vals = isointerp_MiH(Mi_vals)
            J_vals = isointerp_MiJ(Mi_vals)
            K_vals = isointerp_MiK(Mi_vals)
            weight = functionIMFKoupra(Mi_vals)
            isoarr = np.column_stack((Mi_vals, J_vals, H_vals, K_vals, weight))

            isodf = pd.DataFrame(isoarr, columns=['mass_i', 'Jabs', 'Habs', 'Kabs', 'weight'])

            iso_info['isodict'][interpname] = isodf


        sys.stdout.write(str(i)+' / '+str(len(isoage))+'\r')
        i += 1


    iso_pickle = '../Data/Isochrones/isochrone_distributions_resampled.pickle'
    with open(iso_pickle, 'wb') as handle:
        pickle.dump(iso_info, handle)