"""
WHOLE GALAXY DISTRIBUTION FUNCTION
"""
import agama

class GalDF:

    ## CLASS CONSTRUCTOR
    #  thindiskdf_par    - dictionary of thin disc parameters
    #  thickdiskdf_par   - dictionary of thick disc parameters
    #  stellarhalodf_par - dictionary of stellar halo parameters
    #  galpot_par        - list of dictionaries of potential parameters for 
    #                      thin disk, thick disk, gas disk, and dark matter halo
    #  fthick            - weight on thick disc
    #  fhalo             - weight on stellar halo 
    def  __init__(self,galpot_par,thindiskdf_par,thickdiskdf_par,stellarhalodf_par,fthick,fhalo):
        self.galpot_par        = galpot_par
        self.thindiskdf_par    = thindiskdf_par
        self.thickdiskdf_par   = thickdiskdf_par
        self.stellarhalodf_par = stellarhalodf_par
        self.fthick            = fthick
        self.fhalo             = fhalo
        
        ## GALACTIC POTENTIAL
        print(galpot_par)
        gp = agama.Potential(galpot_par[0],galpot_par[1],galpot_par[2],galpot_par[3],galpot_par[4]) 
        
        ## THIN DISC DISTRIBUTION FUNCTION        
        tau1 = 0.01 # Gyr
        taum = 10.  # Gyr
        thindiskdf_dict = dict(
            type       = 'PseudoIsothermal',
            Sigma0     = 1.0,                      # surface density normalization (value at R=0)
            Rdisk      = thindiskdf_par[0],        # scale radius of the (exponential) disk surface density
            Jphimin    = thindiskdf_par[1],        # lower cutoff for evaluating epicyclic frequencies: take max(Jphi,Jphimin)
            Jphi0      = thindiskdf_par[2],        # scale angular momentum determining the suppression of retrograde orbits
            sigmar0    = thindiskdf_par[3],        # normalization of radial velocity dispersion at R=0
            sigmaz0    = thindiskdf_par[4],        # normalization of vertical velocity dispersion at R=0
            sigmamin   = thindiskdf_par[5],        # lower limit on the radial velocity dispersion (because otherwise becomes razor thin): take max(sigmar,sigmamin)
            Rsigmar    = thindiskdf_par[6],        # scale radius of radial velocity dispersion: sigmar=sigmar0*exp(-R/Rsigmar)
            Rsigmaz    = thindiskdf_par[7],        # scale radius of vertical velocity dispersion (default for both should be 2*Rdisk)
            beta       = 0.33,                     # factor describing the growth of velocity dispersion with age
            Tsfr       = 8.,                       # timescale for exponential decline of star formation rate in units of galaxy age
            sigmabirth = (tau1/(taum+tau1))**0.33, # ratio of velocity dispersion at birth to the one at maximum age
            pot        = gp)
        print(thindiskdf_dict)
        thnddf = agama.DistributionFunction(potential = gp, **thindiskdf_dict)
        thnddf_mass = thnddf.totalMass()

        ## THICK DISC DISTRIBUTION FUNCTION (NO AGE-VELOCITY DISPERSION RELATION)     
        thickdiskdf_dict = dict(
            type       = 'PseudoIsothermal',
            Sigma0     = 1.0,
            Rdisk      = thickdiskdf_par[0],
            Jphimin    = thickdiskdf_par[1],
            Jphi0      = thickdiskdf_par[2],
            sigmar0    = thickdiskdf_par[3], 
            sigmaz0    = thickdiskdf_par[4], 
            sigmamin   = thickdiskdf_par[5],   
            Rsigmar    = thickdiskdf_par[6],
            Rsigmaz    = thickdiskdf_par[7],
            pot        = gp)
        thkddf = agama.DistributionFunction(potential = gp, **thickdiskdf_dict)
        thkddf_mass = thkddf.totalMass()
    
        ## HALO DISTRIBUTION FUNCTION   
        stellarhalodf_dict = dict(    
            type      = 'DoublePowerLaw',
            Jcutoff   = stellarhalodf_par[0], # cutoff action (sets exponential suppression at J>Jcutoff, 0 to disable)
            slopeIn   = stellarhalodf_par[1], # power-law index for actions below the break action (Gamma)
            slopeOut  = stellarhalodf_par[2], # power-law index for actions above the break action (Beta)
            steepness = stellarhalodf_par[3], # steepness of the transition between two asymptotic regimes (eta)
            coefJrIn  = stellarhalodf_par[4], # contribution of radial   action to h(J), controlling anisotropy below J_0 (h_r)
            coefJzIn  = stellarhalodf_par[5], # contribution of vertical action to h(J), controlling anisotropy below J_0 (h_z)
            coefJrOut = stellarhalodf_par[6], # contribution of radial   action to g(J), controlling anisotropy above J_0 (g_r)
            coefJzOut = stellarhalodf_par[7], # contribution of vertical action to g(J), controlling anisotropy above J_0 (g_z)
            J0        = stellarhalodf_par[8], # break action (defines the transition between inner and outer regions)
            norm      = 1.)                   # normalization factor with the dimension of mass
        shdf = agama.DistributionFunction(potential = gp, **stellarhalodf_dict)
        shdf_mass = shdf.totalMass()

        # Share variables with class        
        self.thnddf      = thnddf
        self.thnddf_mass = thnddf_mass
        self.thkddf      = thkddf
        self.thkddf_mass = thkddf_mass
        self.shdf        = shdf
        self.shdf_mass   = shdf_mass
    
    ## TOTAL DISTRIBUTION FUNCTION
    # Returns probability
    def __call__(self,acts):
        prob = self.thnddf(acts)/self.thnddf_mass + \
               self.fthick*self.thkddf(acts)/self.thkddf_mass + \
               self.fhalo*self.shdf(acts)/self.shdf_mass
    
        return(prob)