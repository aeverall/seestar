import numpy as np
import pandas as pd

import sys, os
os.path.exists("../../Project/Milky/FitGalaxyModels/")
sys.path.append("../FitGalMods/")

from matplotlib import pyplot as plt
import matplotlib

import CoordTrans


def PlotEquatorial(RA,Dec,
                    org=0, 
                    title='', 
                    projection='mollweide', 
                    rad = True, 
                    s=1, c = 'b', a = 1,
                    cmap = 'cool',
                    figsize = (10,5)):

    '''
    PlotEquatorial - Creates a Mollweide plot of the given coordinates

    Parameters
    ----------
        RA: array of floats
            Right ascension in radians (if rad = True) in range [0, 2pi]

        DEC: array of floats
            Declination in radians (if rad = True) in range [-pi, pi]

    **kwargs
    --------
        org: float in range [0,360] - 0
            Coordinate in the middle of the plot (longitudinally)

        title: String - ''
            Title given to the plot

        projection: string - 'mollweide'
            Type of plot - 'mollweide', 'aitoff', 'hammer', 'lambert'

        rad: bool - True
            Angles given in radians(True) or degrees(False)

        s: float [0, 1] - 1
            Size of scatter points

        figsize: tuple of ints - (10,5)
            Size of plot displayed

    Returns
    -------
        None

        Displays the Mollweide plot

    '''

    if rad: x, y = np.degrees(RA), np.degrees(Dec)
    else: x, y = RA, Dec

    x = np.remainder(x+360-org,360) # shift RA values
    x[x>180] -= 360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)#, axisbg ='LightCyan')

    ax.scatter(np.radians(x),np.radians(y), s=s, c=c, alpha=a, cmap = cmap)  # convert degrees to radians

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    ax.set_xticklabels(tick_labels)     # we add the scale on the x axis

    ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Dec")
    ax.yaxis.label.set_fontsize(12)

    ax.grid(True)


def PlotGalactic(l, b, 
                    org=0, 
                    title='', 
                    projection='mollweide', 
                    rad = True, 
                    s=1, c = 'b', a = 1,
                    cmap = 'cool',
                    figsize=(10,5)):

    '''
    PlotGalactic - Creates a Mollweide plot of the given coordinates

    Parameters
    ----------
        l: array of floats
            Galactic longitude in radians (if rad = True) in range [0, 2pi]

        b: array of floats
            Galactic latitude in radians (if rad = True) in range [-pi/2, pi/2]

    **kwargs
    --------
        org: float in range [0,360] - 0
            Coordinate in the middle of the plot (longitudinally)

        title: String - ''
            Title given to the plot

        projection: string - 'mollweide'
            Type of plot - 'mollweide', 'aitoff', 'hammer', 'lambert'

        rad: bool - True
            Angles given in radians(True) or degrees(False)

        s: float [0, 1] - 1
            Size of scatter points

        figsize: tuple of ints - (10,5)
            Size of plot displayed

    Returns
    -------
        None

        Displays the Mollweide plot

    '''

    if rad: x, y = np.degrees(l), np.degrees(b)
    else: x, y = l, b

    x = np.remainder(x+360-org,360) # shift RA values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)#, axisbg ='LightCyan')
    ax.scatter(np.radians(x),np.radians(y), s=s, c=c, alpha = a, cmap = cmap)  # convert degrees to radians

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    ax.set_xticklabels(tick_labels)     # we add the scale on the x axis

    ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("l")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("b")
    ax.yaxis.label.set_fontsize(12)

    ax.grid(True)
