'''
AngleDisks - Functions for analysing the positions and data
             relating to observation plates from a database

Functions
---------

    AngleShift - Converts latitudinal coordinates from range [-pi, pi] to range [pi, 0]
               - Converting from zero in plane to zero at +ve z-axis 

    InvAngleShift - Converts latitudinal coordinates from range [pi, 0] to range [-pi, pi]
                  - Converting from zero in plane to zero at +ve z-axis 

    Rotation - Returns same points rotated by a given set of angles
                  
    InverseRotation - Returns same points rotated by a given set of angles
                      towards the z-axis

    PlotGalactic - Creates a Mollweide plot of the given coordinates

    GenerateCircle - Creates a circle of points around a position on a sphere

    EquatToGal - Transformation from equatorial angles (RA, Dec) to Galactic angles (l,b)
               - Same as Payel's CoordTrans.EquatorialToGalactic but only 
                 requiring angles to be transformed.

    EquatorialFields - Plots all plates in catalogue as circles of points in Equatorial coordinates
                     - Plots are on a mollweide projection

    GalacticFields - Plots all plates in catalogue as circles of points in Galactic coordinates
                   - Plots are on a mollweide projection

Requirements
------------
agama

Access to path containing Galaxy modification programes:
    sys.path.append("../FitGalMods/")

    import CoordTrans

DataImpor
ArrayMechanics


'''


import numpy as np
import pandas as pd

import sys, os

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 40})


def AngleShift(Th):

    '''
    AngleShift - Converts latitudinal coordinates from range [-pi, pi] to range [pi, 0]
               - Converting from zero in plane to zero at +ve z-axis 
    Parameters
    ----------
        Th: 1D array of floats (radians [-pi,pi])
                - latitudinal coordinates of points taken from x-y plane

    Returns
    -------
        Th: 1D array of floats (radians [0, pi])
                - latitudinal coordinates taken from z-axis
    '''

    Th = -Th
    Th += np.pi/2
    return Th

def InvAngleShift(x):

    '''
    InvAngleShift - Converts latitudinal coordinates from range [pi, 0] to range [-pi, pi]
                  - Converting from zero in plane to zero at +ve z-axis 
    Parameters
    ----------
        Th: 1D array of floats (radians [0,pi])
                - latitudinal coordinates taken from z-axis

    Returns
    -------
        Th: 1D array of floats (radians [-pi,pi])
                - latitudinal coordinates of points taken from x-y plane
    '''
    x -= np.pi/2
    x = -x
    return x  

def Rotation(Phi, Th, Phi0, Th0):
    '''
    Rotation - Returns same points rotated by a given set of angles
               away from the z-axis

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
    Th = AngleShift(Th)
    Th0 = AngleShift(Th0)
    
    x = np.sin(Th) * np.cos(Phi) * np.cos(Th0) * np.cos(Phi0) + \
        np.sin(Th0) * np.cos(Th) * np.cos(Phi0) - \
        np.sin(Th) * np.sin(Phi) * np.sin(Phi0)
        
    y = np.sin(Th) * np.cos(Phi) * np.cos(Th0) * np.sin(Phi0) + \
        np.sin(Th0) * np.cos(Th) * np.sin(Phi0) + \
        np.sin(Th) * np.sin(Phi) * np.cos(Phi0)
        
    z = - np.sin(Th0) * np.sin(Th) * np.cos(Phi) + \
        np.cos(Th0) * np.cos(Th)
        
    Phi_out = np.arctan(y/x)
    Th_out = np.arccos(z)
    
    Phi_out[x<0] += np.pi
    
    #Th_out[Th_out>np.pi/2] -= np.pi
    Th_out = InvAngleShift(Th_out)
    
    return Phi_out, Th_out

def InverseRotation(Phi, Th, Phi0, Th0):
    '''
    InverseRotation - Returns same points rotated by a given set of angles
                      towards the z-axis

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
    Th = AngleShift(Th)
    Th0 = AngleShift(Th0)
    
    x = np.sin(Th) * np.cos(Phi) * np.cos(Phi0) * np.cos(Th0) + \
        np.sin(Th) * np.sin(Phi) * np.sin(Phi0) * np.cos(Th0) - \
        np.cos(Th) * np.sin(Th0)
        
    y = - np.sin(Th) * np.cos(Phi) * np.sin(Phi0) + \
        np.sin(Th) * np.sin(Phi) * np.cos(Phi0)
        
    z = np.sin(Th) * np.cos(Phi) * np.cos(Phi0) * np.sin(Th0) + \
        np.sin(Th) * np.sin(Phi) * np.sin(Phi0) * np.sin(Th0) + \
        np.cos(Th) * np.cos(Th0)
        
    Phi_out = np.arctan(y/x)
    Th_out = np.arccos(z)
    
    Phi_out[(x<0) & (y>0)] += np.pi
    Phi_out[(x<0) & (y<0)] -= np.pi
    
    #Th_out[Th_out>np.pi/2] -= np.pi
    Th_out = InvAngleShift(Th_out)
    
    return Phi_out, Th_out

def PlotDisk(l, b, ax,
                    org=0, 
                    title='', 
                    rad = True,
                    s=0.5):

    '''
    PlotDisk - Creates a Mollweide plot of the given coordinates

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

    ax.scatter(np.radians(x),np.radians(y), s=s, zorder=-10)  # convert degrees to radians

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


def GenerateCircle(Phi_coord, Th_coord, SolidAngle):

    '''
    GenerateCircle - Creates a circle of points around a position on a sphere

    Parameters
    ----------
        Phi_coord: float (radians [0,2pi])
                - Longitude of the position

        Th_coord: float (radians [0, pi])
                - Latitude of the position

        SolidAngle: float (deg^2)
                - Solid angle contained within generated circle

    Returns
    -------
        Phi: 1D array of floats
                - Longitudinal coordinates of circle around point

        Th: 1D array of floats
                - Latitudinal coordinates of circle around point   
    '''

    N = 30

    #Convert to rad**2
    SolidAngle *= (np.pi/180)**2

    #SA = int[0,2pi][0,delta](sin(t)dtdp)
    #1-cos(delta) = SA/2pi
    #Taylor expand for delta
    delta = np.sqrt (SolidAngle / np.pi)

    Th = np.zeros((N)) + (np.pi/2) - delta
    Phi = np.linspace(0, 2*np.pi, N)

    Phi, Th = Rotation(Phi, Th, Phi_coord, Th_coord)

    return Phi, Th


def EquatToGal(ra, dec):

    '''
    EquatToGal - Transformation from equatorial angles (RA, Dec) to Galactic angles (l,b)
               - Same as Payel's CoordTrans.EquatorialToGalactic but only 
                 requiring angles to be transformed.

    Parameters
    ----------
        ra: 1D array of floats (radians [0, 2pi])
                - Right Ascension of points to be transformed

        dec: 1D array of floats (radians [-pi/2, pi/2])
                - Right Ascension of points to be transformed

    Returns
    -------
        l: 1D array of floats (radians [0, 2pi])
                - l coordinate of points

        b: 1D array of floats (radians [-pi/2, pi/2])
                - b coordinate of points
    '''    

    # Equatorial coordinate system constants
    ragp   = 3.36603292
    decgp  = 0.473477282
    lcp    = 2.145566725

    # ra, dec, s => l,b,s
    cd   = np.cos(dec)
    sd   = np.sin(dec)
    b    = np.arcsin(np.sin(decgp)*sd+np.cos(decgp)*cd*np.cos(ra-ragp))
    l    = lcp-np.arctan2(cd*np.sin(ra-ragp),np.cos(decgp)*sd-np.sin(decgp)*cd*np.cos(ra-ragp))
    l[l<0] = l[l<0] + 2.*np.pi

    return l, b

### GALACTIC (HELIOCENTRIC) TO EQUATORIAL (GALACTOCENTRIC)
def GalToEquat(l,b):

    '''
    GalToEquat - Transformation from Galactic angles (l,b) to Equatorial angles (RA, Dec)
               - Same as Payel's CoordTrans.GalacticToEquatorial but only 
                 requiring angles to be transformed.

    Parameters
    ----------
        l: 1D array of floats (radians [0, 2pi])
                - l coordinate of points

        b: 1D array of floats (radians [-pi/2, pi/2])
                - b coordinate of points

    Returns
    -------
        ra: 1D array of floats (radians [0, 2pi])
                - Right Ascension of points to be transformed

        dec: 1D array of floats (radians [-pi/2, pi/2])
                - Right Ascension of points to be transformed
    '''    

    # Equatorial coordinate system constants
    ragp   = 3.36603292
    decgp  = 0.473477282
    lcp    = 2.145566725

    # l,b,s => ra, dec, s
    cb     = np.cos(b)
    sb     = np.sin(b)
    dec    = np.arcsin(np.cos(decgp)*cb*np.cos(l-lcp)+sb*np.sin(decgp))
    ra     = ragp+np.arctan2(cb*np.sin(lcp-l),sb*np.cos(decgp)-cb*np.sin(decgp)*np.cos(l-lcp))
    ra[ra>2.*np.pi] = ra[ra>2.*np.pi] - 2.*np.pi

        
    return ra, dec

def EquatorialFields(save = False, saven = '', title = ''):

    '''
    EquatorialFields - Plots all plates in catalogue as circles of points in Equatorial coordinates
                     - Plots are on a mollweide projection

    **kwargs
    --------
        save: bool - False
            - Do you want to save the figure

        saven: string - ''
            - If saving, what do you want to name the figure

        title: string - ''
            - What is the title of the figure (to be displayed above)

    Returns
    -------
        None
        produces figure which will be displayed in the terminal or a seperate window
    '''

    df = FbfData()
    df.RArad = df.RAdeg*np.pi/180
    df.DErad = df.DEdeg*np.pi/180

    Phi_plts = np.array(df.RArad)
    Th_plts = np.array(df.DErad)

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='mollweide')

    for i in range(len(df)):
        Phi, Th = GenerateCircle(Phi_plts[i],Th_plts[i],28.3)
        PlotDisk(Phi, Th, ax)

    ax.set_title(title)

    if save: fig.savefig('../Figures/'+saven+'.png')

def GalacticFields(save = False, saven = '', title = ''):

    '''
    GalacticFields - Plots all plates in catalogue as circles of points in Galactic coordinates
                   - Plots are on a mollweide projection

    **kwargs
    --------
        save: bool - False
            - Do you want to save the figure

        saven: string - ''
            - If saving, what do you want to name the figure

        title: string - ''
            - What is the title of the figure (to be displayed above)

    Returns
    -------
        None
        produces figure which will be displayed in the terminal or a seperate window
    '''

    df = FbfData()
    df.RArad = df.RAdeg*np.pi/180
    df.DErad = df.DEdeg*np.pi/180

    df['l'], df['b'] = EquatToGal(df.RArad, df.DErad)

    Phi_plts = np.array(df.l)
    Th_plts = np.array(df.b)

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='mollweide')

    for i in range(len(df)):
        Phi, Th = GenerateCircle(Phi_plts[i],Th_plts[i],28.3)
        PlotDisk(Phi, Th, ax)

    ax.set_title(title)

    if save: fig.savefig('../Figures/'+saven+'.png')

def PlotPlate(Phi, Th, SolidAngle, save = False, saven = '', title = ''):

    '''
    EquatorialFields - Plots all plates in catalogue as circles of points in Equatorial coordinates
                     - Plots are on a mollweide projection

    Parameters
    ----------
        Phi_coord: np.array of floats (radians [0,2pi])
                - Longitude of the position

        Th_coord: np.array of float (radians [0, pi])
                - Latitude of the position

        SolidAngle: float (deg^2)
                - Solid angle contained within generated circle

    **kwargs
    --------
        save: bool - False
            - Do you want to save the figure

        saven: string - ''
            - If saving, what do you want to name the figure

        title: string - ''
            - What is the title of the figure (to be displayed above)

    Returns
    -------
        None
        produces figure which will be displayed in the terminal or a seperate window
    '''

    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111, projection='mollweide')

    for i in range(len(Phi)):
        #print(Phi[i],Th[i])
        Phi_c, Th_c = GenerateCircle(Phi[i],Th[i],SolidAngle)
        PlotDisk(Phi_c, Th_c, ax, s=5)

    #PlotDisk(Star[0], Star[1], ax, s=50)

    ax.set_title(title)


    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family='serif')
    plt.xlabel(r'$l$', fontsize=40, style='italic')
    plt.ylabel(r'$b$', fontsize=40, style='italic')
    plt.tick_params(axis='both', labelsize=30)

    if save: fig.savefig('../Figures/'+saven+'.png')

    return ax