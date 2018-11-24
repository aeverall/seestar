'''


Parameters
----------


**kwargs
--------


Returns
-------


'''

import numpy as np
import pandas as pd
from itertools import product

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

def plotObsSF(interp, realm='SF',
            save=False, saven='', title='',
            scat = False, scatdata=[], 
            fig_given = False, ax = None, fig = None, 
            view_contours = True, ncol = 100, nmag = 120, **kwargs):


    options = {'col_range': interp.SF_colrange,
               'mag_range': interp.SF_magrange}
    options.update(kwargs)
    
    # Array for the coordinates in each dimension
    colmin, colmax = options['col_range'][0], options['col_range'][1]
    magmin, magmax  = options['mag_range'][0], options['mag_range'][1]
    
    # Slight deviation inside boundaries to avoid going outside limits
    colmod  = np.linspace(colmin+1e-4,colmax-1e-4,ncol)
    magmod   = np.linspace(magmin+1e-4,magmax-1e-4,nmag)
    
    options = {'col': colmod,
               'mag': magmod,
               'pointings': []}
    options.update(kwargs)
    
    colmod = options['col']
    magmod = options['mag']

    # Labels and ticks for the grid plots
    axis_ticks = {'col': colmod, 'mag': magmod}
    axis_labels = {'col': r"J - K",
                  'mag': r"H"}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over col, mag, s
    col2d = np.transpose(np.stack([colmod,]*len(magmod)), (1,0))
    mag2d = np.transpose(np.stack([magmod,]*len(colmod)), (0,1))
    
    if realm == 'SF': grid = interp((mag2d,col2d))
    elif realm == 'spectro': grid = interp.SF_interp((mag2d, col2d))
    elif realm == 'photo': grid = interp.DF_interp((mag2d, col2d))
    else: print('realm not correctly specified: SF or spectro or photo')

    # Contourf takes the transpose of the matrix
    grid = np.transpose(grid)
    lvls = np.logspace(-10, np.log10( np.max(grid)*10 ), 20 )
    lvls = lvls[lvls!=np.inf]
    #lvls = np.logspace(-10, 8, 20 )
    #lvls = np.logspace(np.log( np.min( grid ) ), np.log( np.max(grid) ), 20 )
    print(np.max(grid))

    if not fig_given:
        # Set up figure with correct number of plots
        fig = plt.figure(figsize=(10,10))
        print("oldfig")

    if view_contours:
        # Plot the grid on the plotting instance at the given inde
        im = plt.contourf(axis_ticks.get('col'),axis_ticks.get('mag'), grid,
                         colormap='YlGnBu', levels=lvls, norm=LogNorm())

        if not fig_given:
            # Add a colour bar to show the variation over the seleciton function.
            fig.colorbar(im)
            formatter = LogFormatter(10, labelOnlyBase=False) 
            fig.colorbar(im, ticks=[1,5,10,20,50], format=formatter)
        else:
            #formatter = LogFormatter(10, labelOnlyBase=False) 
            cb = fig.colorbar(im, ax=ax, ticks=[1e-10, 1E-5, 1E0, 1E5, 1E10], format='%.0E')# ticks=tks, format=formatter)
            cb.ax.set_yticklabels([r'$10^{-10}$', r'$10^{-5}$', r'$10^{0}$', r'$10^{5}$', r'$10^{10}$'], fontsize=25)

    plt.xlabel(r'$J-K$')
    plt.ylabel(r'$m_H$')
    plt.title(title)

    if scat:
        plt.scatter(scatdata[1], scatdata[0], zorder = 1)

    # If the fig can be saved
    if save: fig.savefig(saven, bbox_inches='tight')


def plotSpectroscopicSF(fieldInts, field,
                        srange = (0.01, 20.), mh = -0.2,
                        continuous=False, nlevels=10, title='',
                        save=False, fname='', **kwargs):

    '''
    plotSFInterpolants - Plots the selection function in any combination of age, mh, s coordinates.
                         Can chose what the conditions on individual plots are (3rd coord, Field...)

    Parameters
    ----------
        fieldInts: DataFrame
                - Database of selection function interpolants for each field in the survey

    **kwargs
    --------
        age, mh, s: 1D array
                - arrays of values which the coordinates will vary over

        fields: list
                - field IDs which the plots will use selection functions from

        Phi, Th: 1D array
                - If var1, var2 == 'l', 'b' - the positions of plates will vary over this range
                - My not be exactly right positions as this depends on the plate position

        pointings: list
                - Database of field pointings - only contains pointings where a selection function
                  interpolant has correctly been produced (photo_bool==True)

    Dependencies
    ------------
        findNearestFields - Returns the nearest field for each point in the given list
                            in angles (smallest angle displacement)

    Returns
    -------
        None

        Plots an axis-shared multi-plot of the specified fields/coordinates selection functions.
    '''

    # Array for the coordinates in each dimension
    smin, smax, ns = srange[0], srange[1], 80
    agemin, agemax, nage = 0.0001, 13, 100
    
    smod    = np.logspace(np.log10(smin),np.log10(smax),ns)
    agemod  = np.linspace(agemin,agemax,nage)
    agemod    = np.logspace(np.log10(agemin),np.log10(agemax),nage)

    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]"}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s2d = np.stack([smod,]*len(agemod))
    age2d = np.transpose(np.stack([agemod,]*len(smod)))
    sf2d = fieldInts.agemhssf.loc[field]((age2d, mh, s2d))

    # Normalise contour levels to see the full range of contours
    gridmax = np.max(sf2d)
    levels = np.linspace(0, gridmax, nlevels)
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    im = plt.contourf(age2d,s2d,sf2d,
                     levels=levels,colormap='YlGnBu')    
    plt.yscale('log')
    plt.xlabel(r'$\tau (Gyr)$')
    plt.ylabel(r'$s (kpc)$')
    plt.title(title)#r"$\mathrm{P}(\mathrm{S}|\mathrm{[M/H] = -0.2},\, s,\, \tau)$")
    plt.colorbar(im, format="%.3f")

    if save:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')

    """
    if not fig_given:
        # Set up figure with correct number of plots
        fig = plt.figure(figsize=(10,10))

    # Plot the grid on the plotting instance at the given inde
    im = plt.contourf(age2d,s2d,sf2d,
                     colormap='YlGnBu', levels=lvls, norm=LogNorm())

    if not fig_given:
        # Add a colour bar to show the variation over the seleciton function.
        fig.colorbar(im)
        formatter = LogFormatter(10, labelOnlyBase=False) 
        fig.colorbar(im, ticks=[1,5,10,20,50], format=formatter)
    else:
        #formatter = LogFormatter(10, labelOnlyBase=False) 
        cb = fig.colorbar(im, ax=ax, ticks=np.linspace(0.01, 0.1, 10), format='%.0E')# ticks=tks, format=formatter)
        #cb.ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1.5}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'], fontsize=25)

    plt.yscale('log')
    plt.xlabel(r'$\tau (Gyr)$')
    plt.ylabel(r'$s (kpc)$')
    plt.title(title)

    # If the fig can be saved
    if save: fig.savefig(fname, bbox_inches='tight')
    """

def plotSpectroscopicSF2(intSF, obsSF, field,
                        srange = (0.01, 20.), mh = -0.2,
                        mass_val=False, mass=None,
                        continuous=False, nlevels=10, title='',
                        save=False, fname='',
                        logage=False, **kwargs):

    # Array for the coordinates in each dimension
    smin, smax, ns = srange[0], srange[1], 80
    agemin, agemax, nage = 0.01, 12., 100
    
    smod    = np.logspace(np.log10(smin),np.log10(smax),ns)
    if logage: agemod    = np.logspace(np.log10(agemin),np.log10(agemax),nage)
    else: agemod  = np.linspace(agemin,agemax,nage)
    

    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]"}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s2d = np.stack([smod,]*len(agemod))
    age2d = np.transpose(np.stack([agemod,]*len(smod)))
    mh2d = np.zeros(np.shape(age2d))+mh
    if mass_val:
        mass2d = np.zeros(np.shape(age2d))+mass
        sf2d = intSF((age2d, mh2d, mass2d, s2d), obsSF[field])     
    else:
        sf2d = intSF((age2d, mh2d, s2d), obsSF[field])

    # Normalise contour levels to see the full range of contours
    gridmax = np.max(sf2d)
    if gridmax>0: levels = np.linspace(0, gridmax, nlevels)
    else: 
        levels = np.linspace(0., 1., nlevels)
        print('SF zero everywhere')
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    im = plt.contourf(age2d,s2d,sf2d,
                     levels=levels,colormap='YlGnBu')    
    plt.yscale('log')
    if logage: plt.xscale('log')
    plt.xlabel(r'$\tau (Gyr)$')
    plt.ylabel(r'$s (kpc)$')
    plt.title(title)#r"$\mathrm{P}(\mathrm{S}|\mathrm{[M/H] = -0.2},\, s,\, \tau)$")
    plt.colorbar(im, format="%.3f")

    if save:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


def plotSFInterpolants(fieldInts, varx, vary, var1, var2, 
                        srange = (0.01, 20.), 
                        continuous=False, nlevels=10, title='',
                        save=False, fname='', **kwargs):

    '''
    plotSFInterpolants - Plots the selection function in any combination of age, mh, s coordinates.
                         Can chose what the conditions on individual plots are (3rd coord, Field...)

    Parameters
    ----------
        fieldInts: DataFrame
                - Database of selection function interpolants for each field in the survey

        (varx, vary): tuple of strings
                - Must be any combination of 'age', 'mh', 's'
                - Determines the variables for the x and y axes of the plots respectively

        var1: string
                - Variable which changes over different columns of plots
                - 'age', 'mh', 's', 'fields', 'l', 'b'

        var2: string
                - Variable which changes over different rows of plots
                - 'age', 'mh', 's', 'fields', 'l', 'b'

    **kwargs
    --------
        age, mh, s: 1D array
                - arrays of values which the coordinates will vary over

        fields: list
                - field IDs which the plots will use selection functions from

        Phi, Th: 1D array
                - If var1, var2 == 'l', 'b' - the positions of plates will vary over this range
                - My not be exactly right positions as this depends on the plate position

        pointings: list
                - Database of field pointings - only contains pointings where a selection function
                  interpolant has correctly been produced (photo_bool==True)

    Dependencies
    ------------
        findNearestFields - Returns the nearest field for each point in the given list
                            in angles (smallest angle displacement)

    Returns
    -------
        None

        Plots an axis-shared multi-plot of the specified fields/coordinates selection functions.
    '''

    # Array for the coordinates in each dimension
    smin, smax, ns        = srange[0], srange[1], 30
    agemin, agemax, nage      = 0.0001, 13, 50
    mhmin, mhmax, nmh       = -2.15, 0.4, 25
    
    smod    = np.logspace(np.log10(smin),np.log10(smax),ns)
    agemod  = np.linspace(agemin,agemax,nage)
    agemod    = np.logspace(np.log10(agemin),np.log10(agemax),nage)
    print(agemod)
    mhmod   = np.linspace(mhmin,mhmax,nmh)
    fields = [4120.0]
    Phi = np.linspace(0, 2*np.pi, 5)
    Th = np.linspace(-np.pi/2, np.pi/2, 5)
    
    options = {'age': agemod,
               'mh': mhmod,
               's': smod,
               'fields': fields,
               'Phi': Phi,
               'Th': Th,
               'pointings': []}
    options.update(kwargs)
    
    agemod = options['age']
    mhmod = options['mh']
    smod = options['s']
    
    # Create list of fields which correspond to given array coordinates
    if (var1, var2) == ('l', 'b'):
        pointings = options['pointings']
        Phi = np.array(options['Phi']).repeat(len(options['Th']))
        Th = np.tile(options['Th'], len(options['Phi']))
        fields = findNearestFields((Phi,Th), pointings, var1, var2)
    else: fields = options['fields']


    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod, 'mh': mhmod,
                  'fields': fields, 'l': options['Phi'], 'b': options['Th']}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]",
                  'fields': 'Field ID'}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s3d = np.transpose(np.stack([[smod,]*len(agemod)]*len(mhmod)), (1,0,2))
    age3d = np.transpose(np.stack([[agemod,]*len(smod)]*len(mhmod)), (2,0,1))
    mh3d = np.transpose(np.stack([[mhmod,]*len(smod)]*len(agemod)), (0,2,1))
    
    # Transposes grids to [x, y, z] = [varx, vary, varz]
    coordinates = {'age':0,'mh':1,'s':2,'fields':3}
    a, b =coordinates[varx], coordinates[vary]
    c = 3-(a+b)
    sf3d = {}
    for fieldID in fields:
        grid = fieldInts.agemhssf.loc[fieldID]((age3d, mh3d, s3d))
        sf3d[fieldID] = grid.transpose((a,b,c))

    # Normalise contour levels to see the full range of contours
    gridmax = np.max([sf3d[x] for x in sf3d])
    levels = np.linspace(0, gridmax, nlevels)
    #levels = np.linspace(0, 1, 10)
    print(gridmax)
    
    # Set up figure with correct number of plots
    nxplots = len(axis_ticks[var1])
    nyplots = len(axis_ticks[var2])
    fig = plt.figure(figsize=(nxplots*8, nyplots*8))
    gs1 = gridspec.GridSpec(nyplots, nxplots)
    gs1.update(wspace=0.0, hspace=0.0) 
    # Dictionary to hold subplot instances so that they can be indexed
    plotdict = {}

    for index in range(nxplots*nyplots):
        x = index%nxplots
        y = index/nxplots
        #Correct gs_index so that we're counting from bottom left, not top left
        gs_index = index+ nxplots*(nyplots-1-2*(index/nxplots))
        
        # For the coordinate systems being plotted, generate grids and plot labels
        if var1 == 'fields':
            # Therefore var2 is age, mh or s
            selgrid = sf3d[fields[x]][:,:,y]
            v1text = 'Field =  '+str(fields[x])
            v2text = axis_labels[var2]+' =  '+str(options[var2][y])
            figtext = v1text+'\n'+v2text
        elif var1 in ('age', 'mh', 's'):
            # Therefore var2 is field
            selgrid = sf3d[fields[y]][:,:,x]
            v1text = axis_labels[var1]+' =  '+str(options[var1][x])
            v2text = 'Field =  '+str(fields[y])
            figtext = v1text+'\n'+v2text
        elif (var1, var2) == ('l','b'):
            selgrid = sf3d[fields[index]][:,:,0]
            l = pointings.loc[fields[index]].l
            b = pointings.loc[fields[index]].b
            v1text = 'l =  '+str.format('{0:.1f}',l*180/np.pi)
            v2text = 'b =  '+str.format('{0:.1f}',b*180/np.pi)
            fieldtext = 'Field =  '+str(fields[index])
            # Print value of static variable
            static = list(set(('age','s','mh'))-set((varx,vary)))[0]
            vtext = axis_labels[static]+' = '+str.format('{0:.1f}',options[static][0])
            figtext = v1text+'\n'+v2text+'\n'+fieldtext+'\n'+vtext
        
        # Contourf takes the transpose of the matrix
        selgrid = np.transpose(selgrid)
        # Normalise the plot so that features are still clearly seen compared with others
        #selgrid = selgrid/selgrid.max()
        
        # List which will contain the ticklabels which will be made invisible on the plot
        ticklabels = []
        
        # If index==0 - both x and y axes should be visible
        if index==0: 
            plotdict[0] = plt.subplot(gs1[gs_index])
            plt.xlabel(axis_labels[varx], **Afont)
            plt.ylabel(axis_labels[vary], **Afont)
        # If only x==0 - y-axis should be shared with index:0
        elif x==0: 
            plotdict[index] = plt.subplot(gs1[gs_index], sharex=plotdict[0])
            ticklabels = plotdict[index].get_xticklabels()
            plt.ylabel(axis_labels[vary], **Afont)
        # If only y==0 - x-axis should be shared with index:0
        elif y==0: 
            plotdict[index] = plt.subplot(gs1[gs_index], sharey=plotdict[0])
            ticklabels = plotdict[index].get_yticklabels()
            plt.xlabel(axis_labels[varx], **Afont)
        # If neither x,y==0 - both axes should be shared with x=0,y[0 plots]
        else:
            sharex = plotdict[x]
            sharey = plotdict[y*nxplots]
            plotdict[index] = plt.subplot(gs1[gs_index], sharex=sharex,
                                                         sharey=sharey)
            ticklabels = plotdict[index].get_yticklabels() + plotdict[index].get_xticklabels() 
        
        # Plot the grid on the plotting instance at the given index
        ax = plotdict[index]

        if continuous:
            im = ax.imshow(selgrid, vmin = levels.min(), vmax = levels.max(),
                             extent = [axis_ticks.get(varx).min(), axis_ticks.get(varx).max(),
                                       axis_ticks.get(vary).min(), axis_ticks.get(vary).max()])  
        else:  
            im = ax.contourf(axis_ticks.get(varx),axis_ticks.get(vary), selgrid,
                             levels=levels,colormap='YlGnBu')    
        # Add text to the plot to show the position of the field and other information
        ax.annotate(figtext, xy=(ax.get_xlim()[1]*0.1, ax.get_ylim()[0]*2), 
                    color='orange', **Afont)
        # Distance scale is set to log space
        if varx == 's': ax.set_xscale('log')
        if vary == 's': ax.set_yscale('log')
        #if varx == 'age': ax.set_xscale('log')
        # Make any labels on plots sharing axes invisible to make the plots more clear
        plt.setp(ticklabels, visible=False)

    fig.suptitle(title)
    # Add a colour bar to show the variation over the seleciton function.
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if save:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')

def plotIntSFDist(intSF, obsSF, field,
                 varx='age', vary='s', z=-0.2, mass=None, 
                srng=(0.01,10.), agerng=(0.01,13.), mhrng=(-2.5,0.5),
                 logx=False, logy=False, nx=100, ny=100,
                nlevels=10, colbar=True, levels=None,
                 title=None, fig=None,
                save=False, fname='', xlab=True, ylab=True, ytick=True, xtick=True,
                Titlefont={'fontname':'serif', 'fontsize':20},
                Axisfont={'fontname':'serif', 'fontsize':40},
                Tickfont = {'fontname':'serif', 'fontsize':40},
                **kwargs):
    
    '''
    plotIntSFDist - Plot the intrinsic selection function distribution as a filled
                    contour over parameter space

    Parameters
    ----------
        intSF - instance of SFInstanceClasses.intrinsicIMFSF class
            Stored in the selection function as SF.instanceIMFSF
        obsSF - instance of SFInstanceClasses.observableSF class
            Stored in the selection function as SF.obsSF
        field - field type
            Label of field to be used for distribution

    **kwargs
    --------
        varx='age', vary='s' - str ('s', 'age' or 'mh')
            - The variables to be used for x and y respectively
        z=-0.2 - float
            - The value of the third variable to be used as a slice
        mass=None - None or float
            - The value of mass to be taken for distribution
            - If None, the IMF is integrated over
        srng=(0.01,10.), agerng=(0.01,13.),mhrng=(-2.5,0.5) - tuple of float
            - The ranges of values to be used for each coordinate if taken as variable
        logx=False, logy=False - bool
            - Use a log scale for x or y respectively
        nx=100, ny=100 - int
            - Number of values for x and y variable to use respectively
        nlevels=10 - int
            - Number of levels of contour to use
        levels=None - None or array of float
            - Levels to be used for contour plot
            - If None, levels are automatically decided
        title=None - None or str
            - Title of plot
            - If None, uses the field label and z coord value as title
        fig=None - None or ax instance
            - Axes for figure to be plotted on
            - If None, a new figure is created
        save=False - bool
            - Do you want to save the figure
        fname='' - str
            - Filename for the saved figure
        xlab=True, ylab=True - bool
            - Label x and y axes with variable name
        xtick=True, ytick=True - bool
            - Display x and y ticks on axis
        Titlefont={'fontname':'serif', 'fontsize':20} - dict
            - Font details for figure title
        Axisfont={'fontname':'serif', 'fontsize':40} - dict
            - Font details for axis labels
        Tickfont = {'fontname':'serif', 'fontsize':40} - dict
            - Font details for axis ticks
    Returns
    -------
        im - contourf instance
            - Can be used for generating a colourbar outside the function
    '''

    variables=['s','age','mh']
    variables.remove(varx)
    variables.remove(vary)
    
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]",
                  'fields': 'Field ID'}
    
    options = {'s': srng,
               'age': agerng,
               'mh': mhrng}
    
    # Array for the coordinates in each dimension
    xmin, xmax, nx = options[varx][0], options[varx][1], nx
    ymin, ymax, ny = options[vary][0], options[vary][1], ny
    
    if logx: xmod = np.logspace(np.log10(xmin),np.log10(xmax),nx)
    else: xmod = np.linspace(xmin,xmax,nx)
    if logy: ymod = np.logspace(np.log10(ymin),np.log10(ymax),ny)
    else: ymod  = np.linspace(ymin,ymax,ny)
    

    # Labels and ticks for the grid plots
    #axis_ticks = {'s': smod, 'age': agemod}

    # Create 3D grids to find values of interpolants over age, mh, s
    x2d = np.stack([xmod,]*len(xmod))
    y2d = np.transpose(np.stack([ymod,]*len(ymod)))
    z2d = np.zeros(np.shape(x2d))+z
    
    values = {varx: x2d,
             vary: y2d,
             variables[0]: z2d}
    
    if mass is not None:
        mass2d = np.zeros(np.shape(x2d))+mass
        sf2d = intSF((values['age'], values['mh'], mass2d, values['s']), obsSF[field])     
    else:
        sf2d = intSF((values['age'], values['mh'], values['s']), obsSF[field])

    # Normalise contour levels to see the full range of contours
    gridmax = np.max(sf2d)
    if gridmax>0: 
        if levels is None: levels = np.linspace(0, gridmax, nlevels)
    else: 
        levels = np.linspace(0., 1., nlevels)
        print('SF zero everywhere')
    
    # Create figure
    if fig is None: fig = plt.figure(figsize=(15, 15))
    else: plt.sca(fig)
    im = plt.contourf(x2d,y2d,sf2d,
                     levels=levels,colormap='YlGnBu')    
    if logy: plt.yscale('log')
    if logx: plt.xscale('log')
        
    if xlab: plt.xlabel(axis_labels[varx], **Axisfont)
    if ylab: plt.ylabel(axis_labels[vary], **Axisfont)
    plt.xticks(**Tickfont)
    plt.yticks(**Tickfont)
    if not xtick: plt.setp(fig.get_xticklabels(), visible=False)
    if not ytick: plt.setp(fig.get_yticklabels(), visible=False)
    
    if title is None: title = "Field {0}, {1} = {2}".format(str(field), axis_labels[variables[0]], str(z))
    plt.title(title, fontdict=Titlefont)
    #r"$\mathrm{P}(\mathrm{S}|\mathrm{[M/H] = -0.2},\, s,\, \tau)$")
    if colbar: plt.colorbar(im, format="%.3f")

    if save:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')
        
    return im



"""
Not sure if these functions are any use any more
"""

def plot1DSF(fieldInts, var, scale = 'lin', **kwargs):

    # Array for the coordinates in each dimension
    smin, smax, ns        = 0.01, 20., 50
    agemin, agemax, nage      = 1, 13, 50
    mhmin, mhmax, nmh       = -2.15, 0.4, 50
    
    if scale == 'log': smod = np.logspace(np.log10(smin),np.log10(smax),ns)
    else: smod = np.linspace(smin,smax,ns)
    agemod  = np.linspace(agemin,agemax,nage)
    mhmod   = np.linspace(mhmin,mhmax,nmh)
    fieldID = 4120.0
    Phi = np.linspace(0, 2*np.pi, 5)
    Th = np.linspace(-np.pi/2, np.pi/2, 5)
    
    options = {'age': agemod,
               'mh': mhmod,
               's': smod,
               'fieldID': fieldID,
               'Phi': Phi,
               'Th': Th,
               'pointings': []}
    options.update(kwargs)
    
    agemod = options['age']
    mhmod = options['mh']
    smod = options['s']


    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod, 'mh': mhmod,
                  'fieldID': fieldID, 'l': options['Phi'], 'b': options['Th']}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]",
                  'fieldID': 'Field ID'}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s3d = np.transpose(np.stack([[smod,]*len(agemod)]*len(mhmod)), (1,0,2))
    age3d = np.transpose(np.stack([[agemod,]*len(smod)]*len(mhmod)), (2,0,1))
    mh3d = np.transpose(np.stack([[mhmod,]*len(smod)]*len(agemod)), (0,2,1))

    sf3d = fieldInts.agemhssf.loc[fieldID]((age3d, mh3d, s3d))
        
    # Transposes grids to [x, y, z] = [varx, vary, varz]
    coordinates = {'age':0,'mh':1,'s':2}
    a=coordinates[var]
    if a==1: b=2
    else: b = 3-(a+1)
    c = 3-(a+b)
    sf3d = sf3d.transpose(b,c,a)
    
    # Set up figure with correct number of plots
    fig = plt.figure(figsize=(10,10))

    plt.plot(axis_ticks[var], sf3d[0,0,:])
    if scale == 'log': plt.xscale('log')
    plt.xlabel(axis_labels[var])
    
    return sf3d[0,0,:]

def plotSumSF(fieldInts, var, scale = 'lin', **kwargs):

    # Array for the coordinates in each dimension
    smin, smax, ns        = 0.001, 20., 50
    agemin, agemax, nage      = 1, 13, 50
    mhmin, mhmax, nmh       = -2.15, 0.4, 50
    
    if scale == 'log': smod = np.logspace(np.log10(smin),np.log10(smax),ns)
    else: smod = np.linspace(smin,smax,ns)
    agemod  = np.linspace(agemin,agemax,nage)
    mhmod   = np.linspace(mhmin,mhmax,nmh)
    fieldID = 4120.0
    Phi = np.linspace(0, 2*np.pi, 5)
    Th = np.linspace(-np.pi/2, np.pi/2, 5)
    
    options = {'age': agemod,
               'mh': mhmod,
               's': smod,
               'fieldID': fieldID,
               'Phi': Phi,
               'Th': Th,
               'pointings': []}
    options.update(kwargs)
    
    agemod = options['age']
    mhmod = options['mh']
    smod = options['s']


    # Labels and ticks for the grid plots
    axis_ticks = {'s': smod, 'age': agemod, 'mh': mhmod,
                  'fieldID': fieldID, 'l': options['Phi'], 'b': options['Th']}
    axis_labels = {'s': r"$s$ (kpc)",
                  'age': r"$\tau$ (Gyr)",
                  'mh': r"[M/H]",
                  'fieldID': 'Field ID'}
    Tfont = {'fontname':'serif', 'weight': 100, 'fontsize':20}
    Afont = {'fontname':'serif', 'weight': 700, 'fontsize':20}

    # Create 3D grids to find values of interpolants over age, mh, s
    s3d = np.transpose(np.stack([[smod,]*len(agemod)]*len(mhmod)), (1,0,2))
    age3d = np.transpose(np.stack([[agemod,]*len(smod)]*len(mhmod)), (2,0,1))
    mh3d = np.transpose(np.stack([[mhmod,]*len(smod)]*len(agemod)), (0,2,1))

    sf3d = fieldInts.agemhssf.loc[fieldID]((age3d, mh3d, s3d))
        
    # Transposes grids to [x, y, z] = [varx, vary, varz]
    coordinates = {'age':0,'mh':1,'s':2}
    a=coordinates[var]
    if a==1: b=2
    else: b = 3-(a+1)
    c = 3-(a+b)
    sf3d = sf3d.transpose(b,c,a)
    
    sf = sf3d.sum(axis=0).sum(axis=0)
    
    
    # Set up figure with correct number of plots
    fig = plt.figure(figsize=(10,10))

    plt.plot(axis_ticks[var], sf)
    if scale == 'log': plt.xscale('log')
    plt.xlabel(axis_labels[var])
    
    return sf


def PointDataSample(df, pointings, plate):

    '''
    PointDataSample - Creates sample of points which correspond to a field from spectroscopic catalogue

    Parameters
    ----------
        df: DataFrame
                - Data from the RAVE catalogue with a 'plate' field

        pointings: DataFrame
                - Dataframe of plate coordinates from Wojno's work

        plate: string
                - Id of plate

    Returns
    -------
        points: DataFrame
                - Data about stars on plate including the Theta, Phi coordinates from z-axis

        plate_coord: tuple of floats (radians)
                - Galactic coordinates of centre of the observation plate

    '''

    points = df[df.plate == plate]
    plate_coords = pointings[pointings.plate == plate]
    plate_coords['RArad'], plate_coords['DErad'] = plate_coords.RAdeg*np.pi/180, plate_coords.DEdeg*np.pi/180
    plate_coords['l'], plate_coords['b'] = EquatToGal(plate_coords.RArad, plate_coords.DErad)

    lc, bc = plate_coords.l.iloc[0], plate_coords.b.iloc[0]
    points['Phi'], points['Theta'] = InverseRotation(points.l, points.b, lc, bc)
    points['Th_zero'] = AngleShift(points.Theta)
    plate_coord = (lc,bc)

    points['J_K'] = points.Jmag_2MASS - points.Kmag_2MASS

    return points, plate_coord