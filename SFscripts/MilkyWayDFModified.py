"""
FUNCTION CALCULATING MILKY WAY DF GIVEN OBSERVED COORDINATES. PARAMETERS FROM
http://mnras.oxfordjournals.org/content/445/3/3133.full.pdf (Piffl et al. 2014)
http://mnras.oxfordjournals.org/content/463/3/3169.full.pdf (Das, Williams, and Binney, 2016)

EXAMPLE:
Initialize with solar position, mwdf = MilkyWayDF(solpos)
At equatorial coordinates, we, can calculate actions = mwdf.actions(we), and
distribution function probability prob = mwdf(we)

"""
import numpy as np
import agama
from GalDFModified import GalDF as gdf
import CoordTrans as ct

class MilkyWayDF:
    
    ## CLASS CONSTRUCTOR
    # solpos - Numpy array of Rsun (kpc), zsun (kpc), Usun (km/s), Vsun+VLSR (km/s), Wsun (km/s)
    def  __init__(self,solpos):

        # Share variables
        self.solpos = solpos

        # Set units for agama library
        agama.setUnits(mass=1,length=1,velocity=1)

        # Galaxy potential parameters
        thindisk_par  = dict(type='DiskDensity', 
                             surfaceDensity=5.70657e+08, 
                             scaleRadius=2.6824, 
                             scaleHeight=0.1960)
        thickdisk_par = dict(type='DiskDensity', 
                             surfaceDensity=2.51034e+08, 
                             scaleRadius=2.6824, 
                             scaleHeight=0.7010)
        gasdisk_par   = dict(type='DiskDensity', 
                             surfaceDensity=9.45097e+07,
                             scaleRadius=5.3649, 
                             scaleHeight=0.04,
                             innerCutoffRadius=4.)
        bulge_par     = dict(type='SpheroidDensity',     
                             densityNorm=9.49e+10, 
                             axisRatio=0.5, 
                             gamma=0., 
                             beta=1.8,
                             scaleRadius=0.075,
                             outerCutoffRadius=2.1)
        dmhalo_par    = dict(type='SpheroidDensity', 
                             densityNorm=1.81556e+07, 
                             axisRatio=1.0, 
                             gamma=1., 
                             beta=3.,
                             scaleRadius=14.4336)
        galpot_par    = [thindisk_par,thickdisk_par,gasdisk_par,bulge_par,dmhalo_par]

        # Thin disk distribution function
        Rdisk    = 2.68   
        Rsigmar  = 2.*Rdisk
        Rsigmaz  = 2.*Rdisk                   
        Jphimin  = 0.0                 
        Jphi0    = 100.0                
        sigmar0  = 34.0 * np.exp(solpos[0]/Rsigmar)               
        sigmaz0  = 25.1 * np.exp(solpos[0]/Rsigmaz)  
        sigmamin = 1.0           
        thindiskdf_par = np.array([Rdisk,Jphimin,Jphi0,sigmar0,sigmaz0,sigmamin,Rsigmar,Rsigmaz])
        
        # Thick disk distribution function
        Rdisk    = 2.68                              
        Rsigmar  = 13.0                              
        Rsigmaz  = 4.2                              
        Jphimin  = 0.0
        Jphi0    = 100.0
        sigmar0  = 50.6    
        sigmaz0  = 49.1
        sigmamin = 1.0  
        thickdiskdf_par = np.array([Rdisk,Jphimin,Jphi0,sigmar0,sigmaz0,sigmamin,Rsigmar,Rsigmaz])
        
        # Halo distribution function
        Jcutoff   = 0
        slopeIn   = 2.05
        slopeOut  = 4.72
        steepness = 1.
        coefJrIn  = 1.31
        coefJzIn  = 0.78
        coefJrOut = 1.11
        coefJzOut = 1.29
        J0        = 1635.
        stellarhalodf_par = np.array([Jcutoff,slopeIn,slopeOut,steepness,coefJrIn,coefJzIn,coefJrOut,coefJzOut,J0])
        
        # Weights on thick disc and halo
        fthick = 0.447
        fhalo  = 0.026
            
        # Create potential instance
        gp = agama.Potential(galpot_par[0],galpot_par[1],galpot_par[2],galpot_par[3],galpot_par[4]) 

        # Create action finder instance
        af = agama.ActionFinder(gp)  

        # Create distribution function instance
        from GalDFModified import GalDF as gdf
        galdf = gdf(galpot_par,thindiskdf_par,thickdiskdf_par,stellarhalodf_par,fthick,fhalo)

        # Share with class
        self.af    = af
        self.galdf = galdf
        
    ## ACTIONS 
    # we - Equatorial coordinates (ra (rad), dec (rad), s (kpc), vr (km/s), 
    #                              mura (mas/yr),mudec (mas/yr))
    # Returns actions, Jr, Jz, Jphi
    def actions(self,we):
        
        wg  = ct.EquatorialToGalactic(we)
        wc  = ct.GalacticToCartesian(wg,self.solpos)
        
        # Agama coordinates are right-handed
        wc[:,1] = -wc[:,1]
        wc[:,4] = -wc[:,4]
 
        return(self.af(wc))
    
    def ang_act_coords(self, we):
        
        wg  = ct.EquatorialToGalactic(we)
        wc  = ct.GalacticToCartesian(wg,self.solpos)
        
        # Agama coordinates are right-handed
        wc[:,1] = -wc[:,1]
        wc[:,4] = -wc[:,4]
 
        #actions, angles, frequencies
        return self.af(wc, angles = True)
    
    ## DISTRIBUTION FUNCTION
    # we - Equatorial coordinates (ra (rad),dec (rad),s (kpc),vr (km/s),mura (mas/yr),mudec (mas/yr))
    # Returns probability
    def __call__(self,we):

        acts = self.actions(we)
        prob = self.galdf(acts)
        
        return(prob)