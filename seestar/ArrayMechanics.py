'''
AngleDisks - Functions for analysing the positions and data
             relating to observation plates from a database

Functions
---------
    BoundaryToCentre - converts coordinates for edges of bins to centre of bin coordinates

    BoundaryToCentreNU - BoundaryToCentre for Non-uniform bins

    DensityArray - Takes a series of points in 2D space and converts into a grid of number densities

    ContourPlot - Produces a contour plot from given coordinates

    ParamRangeHighlight - Produces plots selected by specified range in given parameter

    FindRange - Uses a Gaussian fit to determine a suitable range for parameters

    THIS FUNCTION IS NOT CURRENTLY WORKING - NEEDS SOME IMPROVEMENT
    Histogram2D - Should create a 2D grid of populations per grid-square

    Distance - returns the linear distance between 2 points

    AngleSeparation - returns the angular seperation between 2 points

    THIS FUNCTION HAS BEEN REPLATED BY 'Rotation' IN AngleDisks.py
    Distance - returns the angular seperation between 2 points

    PointsToPointings - Adds a column to the df with the number of the field pointing

    PointsToPointingsOverlap - Adds a column to the df with a string of all corresponding field pointings
                             - Includes overlapping field pointings as multiple numbers

    PointsToPointingsMatrix - Adds a column to the df with the number of the field pointing
                            - Uses matrix algebra
                                - Fastest method for asigning field pointings
                                - Requires high memory usage to temporarily hold matrices

    PointsInPointings - Creates a plot with all points which belong to a field pointing
                        - Points are coloured by representative pointing

    AnglePointsToPointingsMatrix - Adds a column to the df with the number of the field pointing
                                 - Uses matrix algebra
                                    - Fastest method for asigning field pointings
                                    - Requires high memory usage to temporarily hold matrices

    PointsToPointingsMatrix - Adds a column to the df with the number of the field pointing
                            - Uses matrix algebra
                                - Fastest method for asigning field pointings
                                - Requires high memory usage to temporarily hold matrices

    EqAreaCircles - Divides a circle of angular extent, delta, into N concentric circles with equal area between circles

    EqAngleSegments - Divides a circle into N-1 equally spaced segments

    EqSections - Divides a range based on a set of points into N equally spaced divisors
               - min and max boundaries are defined by the min/max of the data -/+ a small change (range/(N*5))

    integrateGrid - Performs a trapezium numerical integral over the specified axis

Requirements
------------
agama

Access to path containing Galaxy modification programes:
    sys.path.append("../FitGalMods/")

    import CoordTrans

AngleDisks
'''

import numpy as np
import pandas as pd
import gc
import sys, os
os.path.exists("../../Project/Milky/FitGalaxyModels/")
sys.path.append("../FitGalMods/")

from matplotlib import pyplot as plt
import matplotlib

from seestar import AngleDisks


def BoundaryToCentre(bin_bounds):

    '''
    BoundaryToCentre - converts coordinates for edges of bins to centre of bin coordinates

    Parameters
    ----------
        bin_bounds: 1d np array  
                    Edge coordinates of each bin

    Returns
    -------
        bin_centres: 1d np narray of length 1 less than bin_bounds
                    Coordinates at centre of bin
    '''

    half_width = ((bin_bounds[1]-bin_bounds[0])/2)
    bin_centers = bin_bounds[:len(bin_bounds)-1] + half_width
    return bin_centers

def BoundaryToCentreNU(bin_bounds):

    '''
    BoundaryToCentreNU - BoundaryToCentre for Non-uniform bins

    Parameters
    ----------
        bin_bounds: 1d np array  
                    Edge coordinates of each bin

    Returns
    -------
        bin_centres: 1d np narray of length 1 less than bin_bounds
                    Coordinates at centre of bin
    '''

    bin_centers = np.zeros((len(bin_bounds)-1))
    for i in range(len(bin_bounds)-1):
        bin_centers[i] = (bin_bounds[i] + bin_bounds[i+1])/2


    return bin_centers


def DensityArray(data_frame, x_label, y_label, x_range, y_range, bins):

    '''
    DensityArray - Takes a series of points in 2D space and converts into a grid of number densities

    Parameters
    ----------
        data_frame: pandas DataFrame
                    df including all coordinates for stars in sample

        x_label, y_label: string, string - 
                    coordinates used for x and y axes of grid, 
                    same as column headers in dataframe

    **kwargs
    --------

        x_range, y_range: tuples of floats - default FindRange
                    (xmin, xmax), (ymin, ymax) - x_limits and y_limits of grid
                    FindRange - function determines a suitible range for parameters based on a Gaussian fit

        bins: int
                    number of bins per axis used in contour plot

    Returns
    -------
    return num_grid, x, y

        num_grid: 2d array of same length as x, y
                    number of points per grid square
    
        x, y: 1d np arrays
                    x_coordinates and y_coordinates of grid
    '''

    bins_x = np.linspace(x_range[0],
                         x_range[1],
                         bins)
    bins_y = np.linspace(y_range[0],
                         y_range[1],
                         bins)
    
    num_grid, x_bounds, y_bounds = np.histogram2d(x = getattr(data_frame, x_label),
                                                   y = getattr(data_frame, y_label),
                                                   bins = [bins_x, bins_y])

    x = BoundaryToCentre(x_bounds)
    y = BoundaryToCentre(y_bounds)
    num_grid = np.transpose(num_grid)
    
    return num_grid, x, y


def ContourPlot(df, x_label, y_label, **kwargs):

    '''
    ContourfPlot - Produces a contour plot from given coordinates

    Parameters
    ----------
        data: pandas DataFrame
                data of stars

        x_label: string
            x coordinate of plot

        y_label: string- 
            y coordinate of plot

    Returns
    ------
        None
    '''
    
    options = {'cont_levels': np.linspace(1, 1000, 100),
               'x_range': FindRange(df, x_label, Plot = False),
               'y_range': FindRange(df, y_label, plot = False),
               'bins': 30,
               'save': False,
               'save_name': 'Unknown'}
    options.update(kwargs)
    
    units = 'km/s'
    grid, x, y = DensityArray(df, x_label, y_label,
                                x_range = options['x_range'],
                                y_range = options['y_range'],
                                bins = options['bins'])
    
    plt.contourf(x, y, grid, 
                levels = options['cont_levels'],
                cmap = 'Blues')
    plt.xlabel(x_label + ' ('+units+')')
    plt.ylabel(y_label + ' ('+units+')')
    
    if options['save']:
        fig.savefig('../Figures/' + options['save_name'])



def ParamRangeHighlight(df, parameter, param_range, **kwargs):

    '''
    ParamRangeHighlight - Produces plots selected by specified range in given parameter

    Parameter
    ------
        df: pandas DataFrame
            Data on stars being analysed
        
        parameter: string
            Column header in df of parameter imposing filter
        
        param_range: tuple of doubles
            Max and min value of parameter to be plotted
    
    **kwargs
    --------
        save: bool - default = False
            Change to True if you want to save the figure
    
        save_name: string - default = 'Unknown'
            Enter name of file to save as within Research2017/Figures/ 

    Returns
    ------
        None
    '''

    options = {'save': False,
               'save_name': 'Unknown'}
    options.update(kwargs)
    
    x_label = 'Jphi'
    y_label = 'Jr'
    
    colours = np.ones((len(getattr(df, parameter))))
    colours[(getattr(df, parameter) > param_range[1]) | \
            (getattr(df, parameter) < param_range[0])] = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    title = parameter + ':_' + \
            '{0:.3}'.format(param_range[0]) + \
            '_:_' + \
            '{0:.3}'.format(param_range[1])
    fig.suptitle(title)
    
    plt.sca(axes[0])
    x, y = getattr(df, x_label), getattr(df, y_label)
    plt.scatter(x, y, 
                s = 0.5,
                c = colours,
                cmap = 'Blues')
    plt.xlim(1.4, 2.6)
    plt.ylim(0., .1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    df_filtered = df[(getattr(df, parameter) > param_range[0]) & \
                     (getattr(df, parameter) < param_range[1])]
    
    plt.sca(axes[1])
    ContourPlot(df_filtered, x_label, y_label,
                x_range = (1.4, 2.6), y_range = (0., 0.1),
                cont_levels = np.linspace(1, 300, 80))
    
    plt.sca(axes[2])
    ContourPlot(df_filtered, 'U', 'V',
                x_range = (-80, 40), y_range = (-80, 50),
                cont_levels = np.linspace(1, 300, 80))

    if options['save']:
        fig.savefig('../Figures/' + options['save_name'])



def FindRange(df, Coordinate, **kwargs):

    '''
    FindRange - Uses a Gaussian fit to determine a suitable range for parameters

    Parameters
    ----------
        df: pandas DataFrame
                    Data of stars in sample

        Coordinate: string 
                    Column header in df

    **kwargs
    --------
        Plot: bool
                    Determines whether histograms are printed out for analysis

    Returns
    -------
    return x_limits

        x_limits: tuple of floats
                    return range over which Coordinate contains significant numbers of stars

    '''

    options = {'Plot': False}
    options.update(kwargs)

    Values = getattr(df, Coordinate)
    Values = Values[pd.notnull(Values)]
    num, bin_bound = np.histogram(Values, bins = 2000)

    #Array of histogram bin mid-points
    half_width = ((bin_bound[1]-bin_bound[0])/2)
    x = bin_bound[:len(num)] + half_width

    #Gaussian curve fitting to graph
    from scipy.optimize import curve_fit
    def Gaussian(x, a, mu, s):
        p = a*(1/np.sqrt(2*(s**2)*np.pi)) * np.exp(-((x-mu)**2)/(2*(s**2)))
        return p
    mean = np.sum(x*num)/np.sum(num)
    sigma = np.sum(num*x**2)/np.sum(num)

    #RuntimeError occurs when a Gaussian cannot be fitted to the distribution
    try:
        popt,pcov = curve_fit(Gaussian,x,num,p0=[1e5,mean,sigma])
        perr = np.sqrt(np.diag(pcov))
        np.abs(popt[2]), perr[2], popt[1], perr[1]

        std_dev = np.abs(popt[2])
        fit_mean = popt[1]
        x_limits = (fit_mean-3*std_dev, fit_mean+3*std_dev)

        if options['Plot']:
            myfig = plt.figure(figsize=(10,10))
            ax = myfig.add_subplot(111)
            #Create Gaussian distribution to be added to the plot
            x_vals = np.arange(-4000, 4000, 1)
            y_vals = Gaussian(x_vals, popt[0], popt[1], popt[2])

            ax.plot(x_vals, y_vals, 'r-', label = 'Gaussian Fit')
            ax.plot(x, num, 'b', label = 'Real Data')
            ax.set_xlim(x_limits)
            ax.legend(loc = 'upper right', fontsize = 'large')

    #Take full data range when Gaussian cannot be fitted
    except RuntimeError:
        x_limits = (np.min(Values), np.max(Values))
    
    return x_limits



# N.B. it didn't work and this is pretty crap
def Histogram2D(x_data, y_data, **kwargs):

    '''
    THIS FUNCTION IS NOT CURRENTLY WORKING - NEEDS SOME IMPROVEMENT
    Histogram2D - Should create a 2D grid of populations per grid-square

    Parameters
    ----------
        x_data: pandas Series or numpy 1D Array
                    x coordinates of all stars in sample
        y_data: pandas Series or numpy 1D Array
                    y coordinates of all stars in sample
    **kwargs
    --------
        
        xbounds: tuple of doubles
                    upper and lower limit of grid in x axis
        ybounds: tuple of doubles
                    upper and lower limit of grid in y axis

        xbins: int
                    number of grid squares along x-axis
        ybins: int
                    number of grid squares along y-axis

    Returns
    -------
    return num_map, x_bounds, y_bounds

        num_map: numpy 2D array
                    number of stars per grid square

        x_bounds: numpy 1D array
                    edge coordinates of all histogram bins in x-axis
        y_bounds: numpy 1D array
                    edge coordinates of all histogram bins in y-axis

    '''
    
    options = {'xbounds': (np.min(x_data), np.max(x_data)),
               'ybounds': (np.min(y_data), np.max(y_data)),
               'xbins': 10,
               'ybins': 10}
    options.update(kwargs)
    
    xbins, ybins = options['xbins'], options['ybins']
    
    xy_data = pd.DataFrame(np.transpose(np.array((x_data, y_data))),
                             columns = ['x', 'y'])
    
    x_bounds = np.linspace(options['xbounds'][0], 
                           options['xbounds'][1], 
                           xbins)
    y_bounds = np.linspace(options['ybounds'][0], 
                           options['ybounds'][1], 
                           ybins)
    num_map = np.zeros((xbins-1, 
                        ybins-1))
    
    for i in range(xbins-1):
        for j in range(ybins-1):
            num_map[i,j] = len(xy_data[(xy_data.x>x_bounds[i]) & \
                                       (xy_data.x<x_bounds[i+1]) & \
                                       (xy_data.y>y_bounds[j]) & \
                                       (xy_data.y>x_bounds[j+1])])
    
    return num_map, x_bounds, y_bounds


def Distance(x, y, x0, y0):
    '''
    Distance - returns the linear distance between 2 points

    Parameters
    ----------
        x: float or ndarray of floats
            x-coordinate of point1

        y: float or ndarray of floats
            y-coordinate of point1

        x0: float or ndarray of floats
            x-coordinate of point2

        y0: float or ndarray of floats
            y-coordinate of point2

    Returns
    -------
    return d
        d: float or ndarray of floats
            straight line distance between the 2 points

    N.B. all ndarrays must be the same size
    '''
    d = ((x-x0)**2 + (y-y0)**2)**0.5
    return d

def AngleSeparation(Phi, Th, Phi0, Th0):
    '''
    AngleSeparation - returns the angular seperation between 2 points

    Parameters
    ----------
        Th: float or ndarray of floats - [-pi, pi]
            Theta coordinate of point 1

        Phi: float or ndarray of floats - [0, 2pi]
            Phi coordinate of point1

        Th0: float or ndarray of floats - [-pi, pi]
            Theta coordinate of point 2

        Phi0: float or ndarray of floats - [0, 2pi]
            Phi coordinate of point2

    Returns
    -------
    return delta
        delta: float or ndarray of floats
            direct angle between 2 points

    N.B. all ndarrays must be the same size
    '''

    Th = AngleDisks.AngleShift(Th)
    Th0 = AngleDisks.AngleShift(Th0)

    delta = np.arccos( np.cos(Th0)*np.cos(Th) + \
                       np.sin(Th0)*np.sin(Th)*np.cos(Phi - Phi0))

    return delta

def RotationIncorrect(Phi, Th, Phi0, Th0):
    '''
    THIS FUNCTION HAS BEEN REPLATED BY 'Rotation' IN AngleDisks.py
    Distance - returns the angular seperation between 2 points

    Parameters
    ----------
        Th: float or ndarray of floats - [-pi, pi]
            Theta coordinate of points based around z axis

        Phi: float or ndarray of floats - [0, 2pi]
            Phi coordinate of points based around z axis

        Th0: float or ndarray of floats - [-pi, pi]
            Theta coordinate point which z axis is rotated to

        Phi0: float or ndarray of floats - [0, 2pi]
            Phi coordinate point which z axis is rotated to

    Returns
    -------
    return Phi_out, Th_out
        Phi_out: float or ndarray of floats
            direct angle between 2 points

        Th_out:

    N.B. all ndarrays must be the same size
    '''

    x = np.sin(Th) * np.cos(Phi) * np.cos(Th0) * np.cos(Phi0) - \
        np.sin(Th0) * np.cos(Th) * np.cos(Phi0) - \
        np.sin(Th) * np.sin(Phi) * np.sin(Phi0)

    y = np.sin(Th) * np.cos(Phi) * np.cos(Th0) * np.sin(Phi0) - \
        np.sin(Th0) * np.cos(Th) * np.sin(Phi0) + \
        np.sin(Th) * np.sin(Phi) * np.cos(Phi0)

    z = np.sin(Th0) * np.sin(Th) * np.cos(Phi) + \
        np.cos(Th0) * np.cos(Th)

    Phi_out = np.arctan(y/x)
    Th_out = np.arccos(z)

    return Phi_out, Th_out


def PointsToPointings(df, pointings):

    '''
    PointsToPointings - Adds a column to the df with the number of the field pointing

    Parameters
    ----------
        df: pd.DataFrame
            Contains an x and y column corresponding to the coordinates of points on the contingent axes (RA,Dec)

        pointings: pd.DataFrame
            Contains an x, y, and r column corresponding to positions and radii of field pointings

    Returns
    -------
        df: pd.DataFrame
            Same as input df with:
                - a bool column added for each field pointing
                - a column added which provides the number of the field pointing

    '''
    
    df['point'] = -1
    
    point = np.zeros(len(df)) - 1

    for i in range(len(pointings)):
        df['bool'+str(i)] = Distance(df.x, df.y,
                                     pointings.x.loc[i], 
                                     pointings.y.loc[i]) < pointings.r.loc[i]
        
        point[df['bool'+str(i)]] = i
    df.point = point
    
    return df
    
def PointsToPointingsOverlap(df, pointings):

    '''
    PointsToPointingsOverlap - Adds a column to the df with a string of all corresponding field pointings
                             - Includes overlapping field pointings as multiple numbers

    Parameters
    ----------
        df: pd.DataFrame
            Contains an x and y column corresponding to the coordinates of points on the contingent axes (RA,Dec)

        pointings: pd.DataFrame
            Contains an x, y, and r column corresponding to positions and radii of field pointings

    Returns
    -------
        df: pd.DataFrame
            Same as input df with:
                - a bool column added for each field pointing
                - a column added which provides the numbers of all possible field pointings

    '''
    
    df['point'] = -1

    for i in range(len(pointings)):
        df['bool'+str(i)] = Distance(df.x, df.y,
                                     pointings.x.loc[i], 
                                     pointings.y.loc[i]) < pointings.r.loc[i]
        
        point_i = np.zeros(len(df)).astype(str)
        point_i[:] = ''
        point_i[df['bool'+str(i)]] = str(i)
        df['point'] = np.core.defchararray.add(np.array(df['point']).astype(str), point_i)
    
    return df

def PointsToPointingsMatrix(df, pointings):

    '''
    PointsToPointingsMatrix - Adds a column to the df with the number of the field pointing
                            - Uses matrix algebra
                                - Fastest method for asigning field pointings
                                - Requires high memory usage to temporarily hold matrices

    Parameters
    ----------
        df: pd.DataFrame
            Contains an x and y column corresponding to the coordinates of points on the contingent axes (RA,Dec)

        pointings: pd.DataFrame
            Contains an x, y, and r column corresponding to positions and radii of field pointings

    Returns
    -------
        df: pd.DataFrame
            Same as input df with:
                - a bool column added for each field pointing
                - a column added which provides the number of the field pointing

    '''
    
    df['point'] = -1
    
    point = np.zeros(len(df)) - 1
    
    Mx_df = np.repeat([df.x], len(pointings), axis=0)
    My_df = np.repeat([df.y], len(pointings), axis=0)
    Mx_point = np.transpose(np.repeat([pointings.x], len(df), axis=0))
    My_point = np.transpose(np.repeat([pointings.y], len(df), axis=0))
    Mr_point = np.transpose(np.repeat([pointings.r], len(df), axis=0))
    
    Mbool = Distance(Mx_df,
                     My_df,
                     Mx_point,
                     My_point) < Mr_point
    
    for i in range(len(pointings)):
        point[Mbool[i,:]] = i
        
    df.point = point
    
    return df
    
    
def PointsInPointings(df, pointings):

    '''
    PointsInPointings - Creates a plot with all points which belong to a field pointing
                        - Points are coloured by representative pointing

    Parameters
    ----------
        df: pd.DataFrame
            Contains an x and y column corresponding to the coordinates of points on the contingent axes (RA,Dec)

        pointings: pd.DataFrame
            Contains an x, y, and r column corresponding to positions and radii of field pointings

    Returns
    -------
        None

        Plots points which are near enough to pointings

    '''
    
    def plotting(condition):
        df_conditioned = df[condition]
        plt.scatter(df_conditioned.x, df_conditioned.y)
        plt.xlim(0,1)
        plt.ylim(0,1)

    fig = plt.figure(figsize=(10,10))
    for i in range(len(pointings)):
        plotting(df['bool'+str(i)])

def AnglePointsToPointingsMatrix(df, pointings, Phi, Th, halfangle, 
                                IDtype = str, Nsample = 10000, 
                                progress=False, outString=""):

    '''
    AnglePointsToPointingsMatrix - Adds a column to the df with the number of the field pointing
                                 - Uses matrix algebra
                                    - Fastest method for asigning field pointings
                                    - Requires high memory usage to temporarily hold matrices

    Parameters
    ----------
        df: pd.DataFrame
            Contains Theta and Phi column corresponding to the coordinates of points on the contingent axes (RA,Dec)

        pointings: pd.DataFrame
            Contains an x, y, and r column corresponding to positions and radii of field pointings

        Th: string
            Column header for latitude coordinate (Dec or b)

        Phi: string
            Column header for longitude coordinate (RA or l)

        halfangle: string
            Column header for half-angle of plate on sky

    kwargs
    ------
        IDtype: object
            Type of python object used for field IDs 

        Nsample: int
            Number of stars to be assigned per iterations
            Can't do too many at once due to computer memory constraints

        progress=False: bool
            If true, give a running update of how much of the dataframe has been iterated

        overlap=False: bool
            If true, output field overlap information in DataFrame

    Returns
    -------
        df: pd.DataFrame
            Same as input df with:
                - a bool column added for each field pointing
                - a column added which provides the number of the field pointing
    '''
    df = df.copy()
    if 'points' in list(df): df.drop('points', axis=1, inplace=True)

    iterated=0

    # Iterate over portions of size, Nsample to constrain memory usage.
    for i in range(int(len(df)/Nsample) + 1):

        dfi = df.iloc[i*Nsample:(i+1)*Nsample]

        iterated +=  len(dfi)
        if progress: sys.stdout.write("\r"+outString+"...Assigning: "+str(iterated)+'/'+str(len(df))+"        ")

        pointings = pointings.reset_index(drop=True)
        pointings = pointings.copy()
        
        Mp_df = np.repeat([getattr(dfi,Phi)], 
                            len(pointings), 
                            axis=0)
        Mt_df = np.repeat([getattr(dfi,Th)], 
                            len(pointings), 
                            axis=0)
        Mp_point = np.transpose(np.repeat([getattr(pointings,Phi)],
                                            len(dfi), 
                                            axis=0))
        Mt_point = np.transpose(np.repeat([getattr(pointings,Th)], 
                                            len(dfi), 
                                            axis=0))
        Msa_point = np.transpose(np.repeat([getattr(pointings,halfangle)], 
                                            len(dfi), 
                                            axis=0))
        # Boolean matrix specifying whether each point lies on that pointing       
        Mbool = AngleSeparation(Mp_df,
                                Mt_df,
                                Mp_point,
                                Mt_point) < Msa_point
        # Clear the matrices in order to conserve memory
        del(Mt_df)
        del(Mp_df)
        del(Mp_point)
        del(Mt_point)
        del(Msa_point)
        gc.collect()

        Plates = pointings.fieldID.astype(str).tolist()
        # Convert Mbool to a dataframe with Plates for headers
        Mbool = pd.DataFrame(np.transpose(Mbool), columns=Plates)
        Mplates = pd.DataFrame(np.repeat([Plates,], len(dfi), axis=0), columns=Plates)
        Mplates = Mplates*Mbool
        # Remove "" entries from the lists
        def filtering(x, remove):
            x = [elem for elem in x if elem!=remove]
            return x
        # Do filtering for fields
        field_listoflists = Mplates.values.tolist()
        field_listoflists = [filtering(x, '') for x in field_listoflists]
        # Clear the matrices in order to conserve memory
        del(Mplates)
        gc.collect()
        # Convert type of all field IDs back to correct type
        field_listoflists = [[IDtype(elem) for elem in row] for row in field_listoflists]   
        # Convert list to series then series to dataframe column
        field_series = pd.Series(field_listoflists)
        field_series = pd.DataFrame(field_series, columns=['points'])
        # Reset index to merge on position then bring index back
        dfi = dfi.reset_index()
        dfi = dfi.merge(field_series, how='inner', right_index=True, left_index=True)
        dfi.index = dfi['index']
        dfi.drop('index', inplace=True, axis=1)

        if i == 0: newdf = dfi
        else: 
            newdf = pd.concat((newdf, dfi))


    return newdf

def EqAreaCircles(delta, N):

    '''
    EqAreaCircles - Divides a circle of angular extent, delta, into N concentric circles with equal area between circles

    Parameters
    ----------
        delta: float
                - Anguler extent of outer most circle

        N: int
                - Total number of circles (including central point)

    Returns
    -------
        conc: array of floats
                - Angular  extent of each of the concentric circles

    '''

    conc = np.zeros((N))
    
    area = delta**2 / (N-1)
    
    conc[0] = 0
    conc[1] = np.sqrt(area)
    for i in range(2, N):
        conc[i] = np.sqrt(i*area)
        
    return conc

def EqAngleSegments(N):

    '''
    EqAngleSegments - Divides a circle into N-1 equally spaced segments

    Parameters
    ----------
        N: int
                - Total number of segments (including 0 and 2pi)

    Returns
    -------
        seg: array of floats
                - Angle of boundary of each segment (radians)
    '''
    
    seg = np.linspace(-np.pi, np.pi, N)
    return seg

def EqSections(datafield, N):

    '''
    EqSections - Divides a range based on a set of points into N equally spaced divisors
               - min and max boundaries are defined by the min/max of the data -/+ a small change (range/(N*5))

    Parameters
    ----------
        datafield: 1D array
                - Data which is going to be divided into the grids created by this function

        N: int
                - Total number of boundaries (including min and max)

    Returns
    -------
        boundaries: array of floats
                - ACoordinates of boundary of each section (radians)
    '''

    data_range = np.max(datafield)-np.min(datafield)
    minimum = np.min(datafield) - (data_range/(N*5))
    maximum = np.max(datafield) + (data_range/(N*5))

    boundaries = np.linspace(minimum,
                             maximum,
                             N)

    return boundaries

def integrateGrid(grid, x, axis):

    '''
    integrateGrid - Performs a trapezium numerical integral over the specified axis

    Parameters
    ----------
        grid: nD array
                - Full multidimensional grid of data

        x: 1d array
                - Coordinates of points along axis which will be integrated

        axis: int
                - Axis of the grid which will be integrated over

    Returns
    -------
        integral: n-1 D array
                - Grid of data with same shape without the axis which has been
                integrated over

    e.g.:
    if shape of grid == (4,6,10,3)
    and axis = 1, length of x must be 6
    shape of integral == (4,10,3)

    '''
    
    # Need to know the dimensions
    nd = len(np.shape(grid))
    
    # Transpose the grid to put integrating axis as 0
    dims = np.arange(0,nd,1).tolist()
    dims.insert(0, dims.pop(axis))
    grid = np.transpose(grid, axes=dims)

    # Trapesium rule takes averages between points
    avgs=(grid[0:len(grid)-1] + grid[1:len(grid)])/2
    deltas = x[1:len(x)] - x[0:len(x)-1]

    # Transpose to put integrating axis at end
    dims = np.arange(0,nd,1).tolist()
    dims.insert(len(dims), dims.pop(0))
    avgs = np.transpose(avgs, axes=dims)

    # Integrate over the axis - leaves the same grid missing 1 axis
    integral = np.sum(deltas*avgs, axis=len(dims)-1)
    
    return integral


def extendGrid(grid, x,
                x_lbound = False, x_ubound = False,
                x_lb = 0., x_ub = 0.,
                axis=0):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    nx = len(x)
    dx = (x[nx-1]-x[0])/(nx-1)
    nD = len(np.shape(grid))

    # Transpose the grid to put integrating axis as 0
    dims = np.arange(0,nD,1).tolist()
    dims.insert(0, dims.pop(axis))
    grid = np.transpose(grid, axes=dims)

    if x_lbound: 
        col_first = np.array([grid[0],])
        x_min = x_lb
    else: 
        col_first = np.zeros((1,)+np.shape(grid[0]))
        x_min = x[0]-dx

    if x_ubound: 
        col_last = np.array([grid[nx-1],])
        x_max = x_ub
    else: 
        col_last = np.zeros((1,)+np.shape(grid[0]))
        x_max = x[len(x)-1]+dx

    grid = np.vstack((col_first, grid, col_last))
    x = np.hstack((x_min, x, x_max))

    # Transpose grid back to normal axes
    dims = np.arange(0,nD,1).tolist()
    dims.insert(axis, dims.pop(0))
    grid = np.transpose(grid, axes=dims)
    
    return grid, x