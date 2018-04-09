'''


Parameters
----------


**kwargs
--------


Returns
-------


'''

class IntrinsicToObservable():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''
    
    def __init__(self):
    
        # Isochrone file location
        self.iso_pickle = '/media/andy/UUI/ExternalData/SFProject/stellarprop_parsecdefault_currentmass.dill'
        # Storage of isochrone information
        self.isoage, self.isomh, self.isodict = (None, None, None)
    
        # Mass scaling relation
        self.scaling = lambda mass, Mmax, Mmin: (mass-Mmin)/(Mmax-Mmin)
        # Mass unscaling relation
        self.unscaling = lambda mass_scaled, Mmax, Mmin: mass_scaled*(Mmax-Mmin) + Mmin
        # Storage of scaled mass values
        self.m_scaled = None
    
        # Conversion of absolute to apparent magnitude
        self.appmag = lambda absmag, s: absmag + 5*np.log10(s*1000/10)
        # Conversion of apparent to absolute magnitude
        self.absmag = lambda appmag, s: absmag - 5*np.log10(s*1000/10)
        
    def __call__(self):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        pass
    
    def Initialise(self):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''
        
        # Import isochrone dictionaries and generate isoage, isomh and isodict entries
        
        isoage, isomh, isodict = ImportIsochrones(self.iso_pickle)
        self.isoage, self.isomh, self.isodict = isoage, isomh, isodict
        
        # Calculate interpolant of min and max mass values from isochrones
        Mmin_interp, Mmax_interp = isoMaxMass(isoage, isomh, isodict)
        self.Mmin_interp = Mmin_interp
        self.Mmax_interp = Mmax_interp        

        # Determine the optimum distribution of scaled masses
        m_scaled = ScaledMasses(isoage, isomh, isodict, self.scaling)
        self.m_scaled = m_scaled        
        
        # Create new isochrone dictionary with scaled mass distribution
        iso_info = Redistribution(isoage, isomh, isodict, m_scaled, self.scaling, self.unscaling)
        self.iso_info = iso_info
        
        # Calculate colour and magnitude interpolants in terms of age, mh, mass scaled
        col_interp, mag_interp = cmInterpolation(isoage, isomh, iso_info, m_scaled)
        self.mag_interp = mag_interp
        self.col_interp = col_interp
    
    def pickleInstance(self, pickle_path):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''
        
        with open(pickle_path, 'wb') as handle:
            pickle.dump((self.Mmax_interp, self.Mmin_interp, 
                         self.col_interp, self.mag_interp), handle)
            
    def LoadInterpolants(self, pickle_path):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''
        
        with open(pickle_path, "rb") as input:
            Mmax_interp, Mmin_interp, col_interp, mag_interp = pickle.load(input)
            
        self.Mmin_interp = Mmin_interp
        self.Mmax_interp = Mmax_interp
        self.mag_interp = mag_interp
        self.col_interp = col_interp
    
    def massScaled(self, age, mh, mass):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''
        
        Mmax = self.Mmax_interp((age, mh))
        Mmin = self.Mmin_interp((age, mh))
        
        # Convert mass to scaled mass for each isochone(interpolated between)
        m_scaled = self.scaling(mass, Mmax, Mmin)
        
        return m_scaled
    
    def ColourMabs(self, age, mh, mass):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''
        
        # Convert mass to scaled mass
        m_scaled = self.massScaled(age, mh, mass)
        
        Colour = self.col_interp((age, mh, m_scaled))
        Mabs = self.mag_interp((age, mh, m_scaled))
        
        return Colour, Mabs
    
    def ColourMapp(self, age, mh, mass, s):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''
        
        Colour, Mabs = self.ColourMabs(age, mh, mass)
        
        Mapp = self.appmag(Mabs, s)
        
        return Colour, Mapp

def ImportIsochrones(iso_pickle):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    # Extract isochrones
    print(iso_pickle)
    print("Undilling isochrone interpolants...")
    with open(iso_pickle, "rb") as input:
        pi = dill.load(input)
    print("...done.\n")

    isoage    = pi.isoage
    isomh     = pi.isomh
    isodict   = pi.isodict
    
    return isoage, isomh, isodict

def isoMaxMass(isoage, isomh, isodict):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    # Scaling interpolation

    agemh_scale = []
    for age in isoage:
        for mh in isomh:
            # Retrieve isochrones from pi
            interpname  = "age"+str(age)+"mh"+str(mh) 
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

def ScaledMasses(isoage, isomh, isodict, scaling):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    Mgrid = np.linspace(0., 1., 500)

    mvals = []

    # For logging progress
    i=0

    for age in isoage:
        for mh in isomh:
            interpname  = "age"+str(age)+"mh"+str(mh) 
            isochrone   = isodict[interpname]

            Mi = isochrone[:,2]
            J = isochrone[:,13]
            H = isochrone[:,14]
            K = isochrone[:,15]
            C = J-K

            # Same for all isochrones
            Mmin = np.min(Mi)
            Mmax = np.max(Mi)
            # Scaled mass varies from 0 to 1
            M_scaled = scaling(Mi, Mmax, Mmin)

            isointerp_MJ = scipy.interpolate.interp1d(M_scaled, J)#, bounds_error=False, fill_value=np.nan)
            isointerp_MH = scipy.interpolate.interp1d(M_scaled, H)#, bounds_error=False, fill_value=np.nan)
            isointerp_MK = scipy.interpolate.interp1d(M_scaled, K)#, bounds_error=False, fill_value=np.nan)
            isointerp_MC = scipy.interpolate.interp1d(M_scaled, C)#, bounds_error=False, fill_value=np.nan)

            # Stagger m in root(col**2+mag**2) intervals
            # Starting list for m values
            mlist = []
            # Starting at minimum mass value
            m=np.min(M_scaled)
            dy = 0.1
            dm=0.001
            while (m+dm)<1.0:
                mlist.append(m)
                dCdm = np.abs(isointerp_MC(m+dm)-isointerp_MC(m))/dm
                dHdm = np.abs(isointerp_MH(m+dm)-isointerp_MH(m))/dm
                dm = dy/np.sqrt(dCdm**2+dHdm**2)
                m+=dm

            # Add m values to the entire list
            mvals.extend(mlist)


        # Update progress on going through age values
        sys.stdout.write(str(i)+' / '+str(len(isoage))+'\r')
        i += 1
    print("")

    mvals =np.array(mvals)


    N=1000
    val = np.linspace(0., 1., N)
    cdf = np.zeros(N)

    for i in range(N):

        cdf[i] = float(np.sum(mvals<val[i]))/len(mvals)

        # Update progress on going through age values
        sys.stdout.write(str(i)+' / '+str(N)+'\r')

    cdfinterp = scipy.interpolate.interp1d(cdf, val)


    Nmass = 500

    m_scaled = cdfinterp(np.random.rand(Nmass))
    m_scaled = np.sort(m_scaled)

    return m_scaled

def Redistribution(isoage, isomh, isodict, 
                   m_scaled, scaling, unscaling):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    iso_info = {}

    # For logging progress
    i=0
    for age in isoage:
        for mh in isomh:
            # Retrieve isochrones from pi
            interpname  = "age"+str(age)+"mh"+str(mh) 
            isochrone   = isodict[interpname]

            # Extract information from isochrones
            Mi = isochrone[:,2]
            J = isochrone[:,13]
            H = isochrone[:,14]
            K = isochrone[:,15]

            # Madimum mass to be used for scaling functions
            Mmax = np.max(Mi)
            Mmin = np.min(Mi)

            # Determine scaled mass
            Mi_scaled = scaling(Mi, Mmax, Mmin)

            # Create isochrone interpolants
            isointerp_MJ = scipy.interpolate.interp1d(Mi_scaled, J)#, bounds_error=False, fill_value=np.nan)
            isointerp_MH = scipy.interpolate.interp1d(Mi_scaled, H)#, bounds_error=False, fill_value=np.nan)
            isointerp_MK = scipy.interpolate.interp1d(Mi_scaled, K)#, bounds_error=False, fill_value=np.nan)

            # Recalculate coordinates using distribution of scaled masses
            Mi = unscaling(m_scaled, Mmax, Mmin)
            J = isointerp_MJ(m_scaled)
            K = isointerp_MK(m_scaled)
            H = isointerp_MH(m_scaled)

            # Place isochrone information in a dataframe
            isoarr = np.column_stack((Mi, J, H, K))
            isodf = pd.DataFrame(isoarr, columns=['mass_i', 'Jabs', 'Habs', 'Kabs'])

            # Write dataframe to isochrone dictionary
            iso_info[interpname] = isodf


        # Update progress on going through age values
        sys.stdout.write(str(i)+' / '+str(len(isoage))+'\r')
        i += 1

    return iso_info

def cmInterpolation(isoage, isomh, iso_info, m_scaled, 
                    agemin = 0,agemax = 13, mhmin=-2.5,mhmax=0.5):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    # Construct mass grid
    massgrid = m_scaled

    # Grids are calculated between age and metallicity ranges
    # Construct age grid
    jagemin    = max(0, 
                     np.sum(isoage<agemin)-1)
    jagemax     = min(len(isoage), 
                      np.sum(isoage<agemax) +1)
    agegrid = isoage[jagemin:jagemax]

    # Construct metallicity grid
    jmhmin     = max(0,
                     np.sum(isomh<mhmin)-1)
    jmhmax     = min(len(isomh),
                     np.sum(isomh<mhmax)+1)
    mhgrid  = isomh[jmhmin:jmhmax]

    # Size of grid in each dimension
    nage    = len(agegrid)
    nmh     = len(mhgrid)
    nmass   = len(massgrid)
    print("nage, nmh, nmass: %d, %d, %d" % (nage, nmh, nmass))

    # MultiIndex dataframe for applying transformations
    index = pd.MultiIndex.from_product([agegrid, mhgrid], names = ['age','mh'])
    age_mh = pd.DataFrame(list(product(agegrid, mhgrid)), columns=['age','mh'])

    # Isochrone string identifiers
    age_mh['isoname'] = "age"+age_mh.age.astype(str)+"mh"+age_mh.mh.astype(str)

    # Absolute magnitude arrays from isodict
    age_mh['absA'] = age_mh.isoname.map(lambda x: iso_info[x].Jabs)
    age_mh['absB'] = age_mh.isoname.map(lambda x: iso_info[x].Kabs)
    age_mh['absC'] = age_mh.isoname.map(lambda x: iso_info[x].Habs)

    # Create colour column and extend masses to full length to include zero values for matrix purposes.
    age_mh['ABcol'] = age_mh.absA - age_mh.absB

    # Restack age_mh to create matrices in colour and magnitude
    age_mh.set_index(['age','mh'], inplace=True)
    absCmat = np.array(age_mh[['absC']].unstack()).tolist()
    absCmat = np.array(absCmat)
    ABcolmat = np.array(age_mh[['ABcol']].unstack()).tolist()
    ABcolmat = np.array(ABcolmat)

    # Expand Colour grids to account for central coordinates
    ABcolmat, agegridCol = SelectionGrid.extendGrid(ABcolmat, agegrid, axis=0, 
                                                    x_lbound=True, x_lb=0.)
    ABcolmat, mhgridCol = SelectionGrid.extendGrid(ABcolmat, mhgrid, axis=1)
    ABcolmat, massgridCol = SelectionGrid.extendGrid(ABcolmat, massgrid, axis=2, 
                                                     x_lbound=True, x_lb=0., x_ubound=True, x_ub=1.)
    # Expand Magnitude grids to account for central coordinates
    absCmat, agegridMag = SelectionGrid.extendGrid(absCmat, agegrid, axis=0, 
                                                   x_lbound=True, x_lb=0.)
    absCmat, mhgridMag = SelectionGrid.extendGrid(absCmat, mhgrid, axis=1)
    absCmat, massgridMag = SelectionGrid.extendGrid(absCmat, massgrid, axis=2, 
                                                    x_lbound=True, x_lb=0., x_ubound=True, x_ub=1.)

    # Interpolate over matrices to get col&mag as a function of age, metallicity, mass
    print((np.min(agegridCol), np.max(agegridCol)), (np.min(mhgridCol), np.max(mhgridCol)), (np.min(massgridCol), np.max(massgridCol)))
    col_interp = RGI((agegridCol, mhgridCol, massgridCol), ABcolmat, bounds_error=False, fill_value=np.nan)
    mag_interp = RGI((agegridMag, mhgridMag, massgridMag), absCmat, bounds_error=False, fill_value=np.nan)

    return col_interp, mag_interp


class mScale():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''
    
    def __init__(self, Mmax_interp, Mmin):
        
        self.Mmax_interp = Mmax_interp
        self.Mmin = Mmin
        
        self.function = lambda mass, Mmax, Mmin: (mass-Mmin)/(Mmax-Mmin)
        
    def __call__(self, age, mh, mass):
        
        mass_scaled = self.function( mass, self.Mmax_interp((age, mh)), self.Mmin)
    
        return mass_scaled
    
class mUnscale():

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''
    
    def __init__(self, Mmax_interp, Mmin):
        
        self.Mmax_interp = Mmax_interp
        self.Mmin = Mmin
        
        self.function = lambda mass_scaled, Mmax, Mmin: mass_scaled*(Mmax-Mmin) + Mmin
        
    def __call__(self, age, mh, mass_scaled):
    
        m = self.function( mass_scaled, self.Mmax_interp((age, mh)), self.Mmin)
        
        return m