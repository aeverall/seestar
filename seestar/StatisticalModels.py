
'''
StatisticalModels.py - Set of functions for building statistical models, calculations and tests.

Classes
-------

Functions
---------

Dependancies
------------


'''

import numpy as np
import pandas as pd
import numpy
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.optimize as op
import sys
from mpmath import *

# Import cubature for integrating over regions
from cubature import cubature


class GaussianMM():

    '''
    GaussianMM - Class for calculating bivariate Gaussian mixture model which best fits
                 the given poisson point process data.

    Parameters
    ----------
        x, y - np.array of floats
            - x and y coordinates for the points generated via poisson point process from 
            the smooth model

        nComponents - int
            - Number of Gaussian components of the mixture model

        rngx, rngy - tuple of floats
            - Upper and lower bounds on x and y of the survey

    Functions
    ---------
        __call__ - Returns the value of the smooth GMM distribution at the points x, y

        optimizeParams - Vary the parameters of the distribution using the given method to optimise the 
                        poisson likelihood of the distribution

        initParams - Specify the initial parameters for the Gaussian mixture model as well as 
                    the lower and upper bounds on parameters (used as prior values in the optimization)

        lnprob - ln of the posterior probability of the distribution given the parameters.
               - posterior probability function is proportional to the prior times the likelihood
               - lnpost = lnprior + lnlike

        lnlike - The poisson likelihood disrtibution of the Gaussian mixture model given the observed points

        lnprior - The test of the parameters of the Gaussian Mixture Model against the specified prior values     

        priorTest - Testing the parameters of the GMM against the upper and lower limits specified in
                    self.initParams

        testIntegral - Test the approximate integral calculated using the given integration rule against the accurate
                        integral calculated using cubature for which we know the uncertainty
    '''
    
    def __init__(self, x, y, nComponents, rngx, rngy):

        # Distribution from photometric survey for maximum prior
        self.photoDF = None
        self.priorDFbool = False
        
        # Number of components of distribution
        self.nComponents = nComponents
        # Starting values for parameters
        self.params_i = []
        # Final optimal values for parameters
        self.params_f = []
        # Min value of parameters as prior
        self.underPriors = []
        # Max value of parameters as prior
        self.overPriors = []
        
        self.x = x
        self.y = y

        self.rngx, self.rngy = rngx, rngy
        
        # Function which calculates the actual distribution
        self.distribution = multiDistribution

        # Print out likelihood values as calculated
        self.runningL = True

        # Values of SD in rotated frame with rho' = 0 - to be used as constraints
        self.sigxtilda = lambda sigx, sigy, r: np.sqrt( ( 2*(sigx**2)*(sigy**2) ) / \
        ( sigx**2 + sigy**2 + np.sqrt((sigy**2 - sigx**2)**2 + (2*r*sigx*sigy)**2)) )
        self.sigytilda = lambda sigx, sigy, r: np.sqrt( ( 2*(sigx**2)*(sigy**2) ) / \
        ( sigx**2 + sigy**2 - np.sqrt((sigy**2 - sigx**2)**2 + (2*r*sigx*sigy)**2)) )
        
    def __call__(self, (x, y)):

        '''
        __call__ - Returns the value of the smooth GMM distribution at the points x, y

        Parameters
        ----------
            x, y - float or np.array of floats
                - x and y coordinates of points at which to take the value of the GMM

        Returns
        -------
            self.distribution(...) - float or np.array of floats
                - The value of the GMM at coordinates x, y
        '''
        
        # Value of coordinates x, y in the Gaussian mixture model
        GMMval = self.distribution(self.params_f, x, y, self.nComponents)

        """
        # Any values outside range - 0
        constraint = (x>=self.rngx[0])&(x<=self.rngx[1])&(y>=self.rngy[0])&(y<=self.rngy[1])
        if (type(GMMval) == np.array)|(type(GMMval) == np.ndarray)|(type(GMMval) == pd.Series): 
            GMMval[~constraint] = 0.
            GMMval[(np.isnan(x))|(np.isnan(y))] = np.nan
        elif (type(GMMval) == float) | (type(GMMval) == np.float64):
            if not constraint: 
                if (np.isnan(x))|(np.isnan(y)): GMMval = np.nan
                else: GMMval = 0.
        else: raise TypeError('The type of the input variables is '+str(type(GMMval)))
        """

        if (type(GMMval) == np.array)|(type(GMMval) == np.ndarray)|(type(GMMval) == pd.Series): 
            # Not-nan input values
            notnan = (~np.isnan(x))&(~np.isnan(y))
            # Any values outside range - 0
            constraint = (x[notnan]>=self.rngx[0])&(x[notnan]<=self.rngx[1])&(y[notnan]>=self.rngy[0])&(y[notnan]<=self.rngy[1])
            GMMval[~notnan] = np.nan
            GMMval[notnan][~constraint] = 0.
        elif (type(GMMval) == float) | (type(GMMval) == np.float64):
            if (np.isnan(x))|(np.isnan(y)): GMMval = np.nan
            else: 
                constraint = (x>=self.rngx[0])&(x<=self.rngx[1])&(y>=self.rngy[0])&(y<=self.rngy[1])
                if not constraint:
                    GMMval = 0.
        else: raise TypeError('The type of the input variables is '+str(type(GMMval)))

        return GMMval
        
        
    def optimizeParams(self, method = "Powell"):

        '''
        optimizeParams - Vary the parameters of the distribution using the given method to optimise the 
                        poisson likelihood of the distribution

        **kwargs
        --------
            method = "Powell" - str
                - The scipy.optimize.minimize method used to vary the parameters of the distribution
                in order to find the minimum negative log likelihooh (min ( -log(L) ))

        Returns
        -------
            nll(result['x']) - float
                - The negative log likelihood for the final solution parameters
        '''

        # Set initial parameters
        self.params_i, self.underPriors, self.overPriors = self.initParams()

        # nll is the negative lnlike distribution
        nll = lambda *args: -self.lnprob(*args)

        # result is the set of theta parameters which optimise the likelihood given x, y, yerr
        result = op.minimize(nll, self.params_i, method = method)
        
        # Save evaluated parameters to internal values
        self.params_f = []
        for i in range(self.nComponents):
            self.params_f.append(result["x"][i*6:(i+1)*6])
       
        return nll(result["x"])

    def initParams(self):

        '''
        initParams - Specify the initial parameters for the Gaussian mixture model as well as 
                    the lower and upper bounds on parameters (used as prior values in the optimization)

        Returns
        -------
            parameters_i - list of floats
                - initial parameters for the Gaussian mixture model
            parameters_u - list of floats
                - "under parameters" - lower limits on the values of GMM parameters
            parameters_o - list of floats
                - "over parameters" - upper limits on the values of GMM parameters
        '''

        # Initial guess parameters for a bivariate Gaussian
        mux_i, muy_i = (self.rngx[0]+self.rngx[1])/2, (self.rngy[0]+self.rngy[1])/2
        sigmax_i, sigmay_i = (self.rngx[1]-self.rngx[0])/10, (self.rngy[1]-self.rngy[0])/10
        rho_i = 0.
        # If calculating SF set initial value so that max = 0.1 at start
        #normalisation = (2 * np.pi * np.abs(sigmax_i * sigmay_i) * np.sqrt(1 - rho_i**2))
        if self.priorDFbool: A_i =  0.1 / self.nComponents 
        else: A_i = 1.

        mux_u, muy_u = self.rngx[0], self.rngy[0]
        sigmax_u, sigmay_u = 0, 0
        A_u = 0.
        rho_u = -1.

        mux_o, muy_o = self.rngx[1], self.rngy[1]
        sigmax_o, sigmay_o = self.rngx[1]-self.rngx[0], self.rngy[1]-self.rngy[0]
        # If calculating SF, A_o cannot be larger than 1
        if self.priorDFbool: A_o = 1.
        else: A_o = np.inf
        rho_o = 1.

        # If smoothing, set values of sigma so that gaussians are sufficiently broad
        smoothing = True
        if smoothing:
            self.smoothFactor = 1/100.
            dx = self.rngx[1]-self.rngx[0]
            dy = self.rngy[1]-self.rngy[0]
            sigmax_u, sigmay_u = (dx*self.smoothFactor, dy*self.smoothFactor)

        p_list = [mux_i, sigmax_i, muy_i, sigmay_i, A_i, rho_i]
        u_list = [mux_u, sigmax_u, muy_u, sigmay_u, A_u, rho_u]
        o_list = [mux_o, sigmax_o, muy_o, sigmay_o, A_o, rho_o]

        # Initial parameters for a Double bivariate Gaussian
        parameters_i = [p_list,]*self.nComponents
        parameters_u = [u_list,]*self.nComponents
        parameters_o = [o_list,]*self.nComponents

        return parameters_i, parameters_u, parameters_o

    def lnprob(self, params):

        '''
        lnprob - ln of the posterior probability of the distribution given the parameters.
               - posterior probability function is proportional to the prior times the likelihood
               - lnpost = lnprior + lnlike

        Parameters
        ----------
            params - list of floats
                - Values of parameters in the Gaussian Mixture model

        Returns
        -------
            lnprior(params)+lnlike(params) - float
                - ln of the posterior probability
                - -np.inf if the prior is false and hence prob=0 - lnprob=-inf
        '''

        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(params)

    # ln(Likelihood) based on a Poisson likelihood distribution
    def lnlike(self, params):

        '''
        lnlike - The poisson likelihood disrtibution of the Gaussian mixture model given the observed points

        Parameters
        ----------
            params - list of floats
                - Values of parameters in the Gaussian Mixture model

        Returns
        -------
            contPoints-contInteg - float
                - lnL for the GMM parameters given the observed points
        '''
        
        param_set = []
        for i in range(self.nComponents):
            param_set.append( params[i*6:(i+1)*6] )

        # If the DF has already been calculated, directly optimise the SF
        if self.priorDFbool: function = lambda (a, b): self.photoDF((a,b)) * self.distribution(param_set, a, b, self.nComponents)
        else: function = lambda (a, b): self.distribution(param_set, a, b, self.nComponents)
        
        # Point component of poisson log likelihood: contPoints \sum(\lambda(x_i))
        model = function((self.x, self.y))
        contPoints = np.sum( np.log(model) )
        
        # Integral of the smooth function over the entire region
        contInteg = integrationRoutine(function, param_set, self.nComponents, (self.rngx, self.rngy))

        lnL = contPoints - contInteg

        if self.runningL:
            sys.stdout.write("\rlogL: %.2f, sum log(f(xi)): %.2f, integral: %.2f            " % (lnL, contPoints, contInteg))
            sys.stdout.flush()
            
        return contPoints - contInteg

    # "uninformative prior" - uniform and non-zero within a specified range of parameter values
    def lnprior(self, params):
        
        '''
        lnprior - The test of the parameters of the Gaussian Mixture Model against the specified prior values

        Parameters
        ----------
            params - list of floats
                - Values of parameters in the Gaussian Mixture model

        Returns
        -------
            lnprior - either 0.0 or -np.inf
                - If the prior is satisfied, p=1. therefore lnp = 0.0
                - If not satisfied, p=0. therefore lnp = -np.inf
        '''

        param_set = []
        for i in range(self.nComponents):
            param_set.append( params[i*6:(i+1)*6] )
        # Test parameters against boundary values
        prior = self.priorTest(param_set)
        # Test parameters against variance
        if prior: prior = prior & self.varTest(param_set)

        # Only calculate if prior is satisfied otherwise bad values of rho arise
        if prior:
            # Prior on spectro distribution that it must be less than the photometric distribution
            if self.priorDFbool:
                function = lambda (a, b): self.distribution(param_set, a, b, self.nComponents)
                prior_df = SFprior(function, (self.rngx, self.rngy))
            else: prior_df = True

            if prior & prior_df:
                return 0.0
            else: 
                return -np.inf

        else: return -np.inf
    
    # Returns a boolean true or false for whether all parameters lie within their range
    def priorTest(self, params):
        
        '''
        priorTest - Testing the parameters of the GMM against the upper and lower limits specified in
                    self.initParams

        Parameters
        ----------
            params - list of floats
                - Values of parameters in the Gaussian Mixture model

        Returns
        -------
             - bool
                - True if parameters satisfy constraints. False if not.
        '''

        Val = np.array(params).flatten()
        minVal = np.array(self.underPriors).flatten()
        maxVal = np.array(self.overPriors).flatten()
        
        minBool = Val > minVal
        maxBool = Val < maxVal
        rngBool = minBool*maxBool

        #if self.priorDFbool: print(Val)
        
        solution = np.sum(rngBool) - len(Val)
        if solution == 0:
            return True
        else: return False

    def varTest(self, params):

        '''
        varTest - tests whether variance is constrained within prior limits

        Parameters
        ----------
            params - list of floats
                - Values of parameters in the Gaussian Mixture model
        Returns
        -------
             - bool
                - True if parameters satisfy constraints. False if not.
        '''

        sigmax, sigmay, rho = np.array(()), np.array(()), np.array(())
        for arr in params:
            sigmax = np.append(sigmax, arr[1])
            sigmay = np.append(sigmay, arr[3])
            rho = np.append(rho, arr[5])

        dx = self.rngx[1]-self.rngx[0]
        dy = self.rngy[1]-self.rngy[0]

        sigxtil = self.sigxtilda(sigmax/dx, sigmay/dy, rho)
        sigytil = self.sigytilda(sigmax/dx, sigmay/dy, rho)

        result = np.sum(sigxtil<self.smoothFactor) + np.sum(sigytil<self.smoothFactor)

        if result == 0: return True
        else: return False


    def testIntegral(self, integration='trapezium'):

        '''
        testIntegral - Test the approximate integral calculated using the given integration rule against the accurate
                        integral calculated using cubature for which we know the uncertainty

        **kwargs
        --------          
            integration='trapezium' - str
                - The type of integration routine to be tested
                - 'analytic', 'trapezium', 'simpson', 'cubature'     

        Returns
        -------
            calc_val - float
                - integral calculated using trapezium rule approximation

            real_val - float
                - integral calculated using cubature routine

            err - float
                - The error of the calculated value with respect to the real value

        Also prints out the values and errors of each calculation automatically
        '''

        function = lambda (a, b): self.distribution(self.params_f, a, b, self.nComponents)

        real_val, err = cubature(function, 2, 1, (self.rngx[0], self.rngy[0]), (self.rngx[1], self.rngy[1]))
        calc_val = integrationRoutine(function, self.params_f, self.nComponents, (self.rngx, self.rngy), integration=integration)

        percent = ((calc_val - float(real_val))/calc_val)*100
        cubature_percent = 100*float(err)/float(real_val)

        print("\nThe error in the linear numerical integral was %.3E%%" % float(percent))
        print("\nThe cubature calculation error is quoted as %.3E or %.3E%%" % (float(err), cubature_percent)  )

        return calc_val, real_val, err


def SFprior(function, (rngx, rngy)):

    '''
    SFprior - The selection function has to be between 0 and 1 everywhere.

    Parameters
    ----------
        function - interp
            - The selection function interpolant over the region, R.

        (rngx, rngy) - tuple of floats
            - range of colours and magnitudes which limit the selection function region.

    Returns
    -------
        prior - bool
            - True if all points on GMM are less than 1.
            - Otherwise False
    '''

    N=150

    x_coords = np.linspace(rngx[0], rngx[1], N)
    y_coords = np.linspace(rngy[0], rngy[1], N)

    x_2d = np.tile(x_coords, ( len(y_coords), 1 ))
    y_2d = np.tile(y_coords, ( len(x_coords), 1 )).T

    SF = function((x_2d, y_2d))

    prior = not np.max(SF)>1

    return prior


def bivariateGauss(params, x, y):

    '''
    bivariateGauss - Calculates the value of the bivariate Gaussian defined by parameters
                    at point x, y

    Parameters
    ----------
        params - list of floats
            - Values of parameters for the Gaussian.

        x, y - float or np.array of floats
            - Coordinates at which the bivariate Gaussian is calculated.

    Returns
    -------
        BG - float or np.array of floats
            - Value of bivariate Gaussian at coordinates
    '''

    mu1, sigma1, mu2, sigma2, A, rho = params
    # Coordinate
    z = ((x - mu1)**2 / sigma1**2) + ((y - mu2)**2 / sigma2**2) + \
        2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
    # When the Series is empty, the data type goes to object so this is corrected:
    z = z.astype(np.float64)
    # Bivariate Gaussian
    if 1-rho**2 == 0:
        print("\nBad rho value:" +str(rho))
    Norm = A#(A/(2 * np.pi * np.abs(sigma1 * sigma2) * np.sqrt(1 - rho**2)))
    Exponent = numpy.exp(-z / (2 * (1 - rho**2)))
    BG = Norm*Exponent
    return BG

def bivariateIntegral(params):

    '''
    bivariateIntegral - Analytic integral over the specified bivariate Gaussian.

    Parameters
    ----------
        params - list of floats
            - Values of parameters for the Gaussian.

    Returns
    -------
        contInteg - float
            - Integral over the bivariate GAussian
    '''

    mux, sigmax, muy, sigmay, A, rho = params
    # Continuous integral of Bivariate Gaussian with infinite boundaries.
    contInteg = 2*np.pi * A * np.abs(sigmax * sigmay) * np.sqrt(1-rho**2)
    return contInteg

def multiDistribution(params, x, y, nComponents):

    '''
    multiDistribution - Value of GMM of bivariate Gaussians at coorindates x, y.

    Parameters
    ----------
        params - list of floats
            - Values of parameters for the Gaussian.

        x, y - float or np.array of floats
            - Coordinates at which the GMM is calculated.

        nComponents - int
            - Number of components of the Gaussian Mixture Model

    Returns
    -------
        p - Value of the GMM at coordinates x, y
    '''

    p = 0
    for i in range(nComponents):
        p += bivariateGauss(params[i], x, y)
    return p

def multiIntegral(params, nComponents):

    '''
    multiIntegral - Analytic integral over the specified bivariate Gaussian.

    Parameters
    ----------
        params - list of floats
            - Values of parameters for the Gaussian.

        nComponents - int
            - Number of components of the Gaussian Mixture Model

    Returns
    -------
        integral - float
            - Integral over the Gaussian Mixture Model
    '''

    integral = 0
    for i in range(nComponents):
        integral += bivariateIntegral(params[i])
    return integral
    
# Gaussian function for generating error distributions
def Gauss(x, mu=0, sigma=1):

    '''
    Gauss - Calculates the value of the 1D Gaussian defined by parameters at point x
    (used for recovering posterior distributions from burnt in Monte Carlo Markov Chains)

    Parameters
    ----------
        x - float or np.array of floats
            - Coordinate at which the Gaussian is calculated.

    **kwargs
    --------
        mu = 0 - float
            - Value of mean of Gaussian
        sigma=1 - float
            - Value of standard deviation of Gaussian

    Returns
    -------
        G - float or np.array of floats
            - Value of Gaussian at coordinate x.
    '''

    G = np.exp(-((x-mu)**2)/(2*sigma**2))
    return G

# Create cumulative distribution from Gauss(x)
def cdf(func, xmin, xmax, N, **kwargs):

    '''
    cdf - Normalised cumulative distribution function of 1D dunction between limits.

    Parameters
    ----------
        func - function or interpolant
            - The 1D function which is being integrated over

        xmin, xmax - float
            - min and max values for the range of the distribution

        N - int
            - Number of steps to take in CDF.

    **kwargs
    --------

    Returns
    -------
        value_interp - interp.interp1d
            - Interpolant of the CDF over specified range.
    '''

    points = np.linspace(xmin, xmax, N)
    
    # Use trapezium rule to calculate the integrand under individual components
    dx = points[1:] - points[:len(points)-1]
    h1 = func(points[:len(points)-1], **kwargs)
    h2 = func(points[1:], **kwargs)
    volumes = dx * ( h2 + h1 ) / 2
    
    # cumulative distribution function = sum of integrands
    cdf = np.zeros_like(volumes)
    for i in range(len(volumes)):
        cdf[i] = np.sum(volumes[:i+1])
    
    # normalisation of cdf
    cdf *= 1/cdf[-1]
    cdf[0] = 0.

    # Linear interpolation of cumulative distribution
    #interpolant = interp.interp1d(points[1:], cdf)
    # Inverse interpolate to allw generation of probability weighted distributions.
    value_interp = interp.interp1d(cdf, points[1:], bounds_error=False, fill_value=np.nan)
    
    return value_interp


def integrationRoutine(function, param_set, nComponents, (rngx, rngy), integration = "trapezium"):

    '''
    integrationRoutine - Chose the method by which the integrate the distribution over the specified region
                        of space then perform the integral.

    Parameters
    ----------
        function - function or interpolant
            - The function to be integrated over the specified region of space

        param_set - list of floats
            - Set of parameters which define the GMM.

        nComponents - int
            - Number of components of the GMM.

        (rngx, rngy) - tuple of floats
            - Boundary of region of colour-magnitude space being calculated.

    **kwargs
    --------
        integration='trapezium' - str
            - The type of integration routine to be tested
            - 'analytic', 'trapezium', 'simpson', 'cubature'    

    Returns
    -------
        contInteg - float
            - Value of the integral after calculation
    '''

    # analytic if we have analytic solution to the distribution - this is the fastest
    if integration == "analytic": contInteg = multiIntegral(param_set, nComponents)
    # trapezium is a simple approximation for the integral - fast - ~1% accurate
    elif integration == "trapezium": contInteg = numericalIntegrate(function, (rngx, rngy))
    # simpson is a quadratic approximation to the integral - reasonably fast - ~1% accurate
    elif integration == "simpson": contInteg = simpsonIntegrate(function, (rngx, rngy))
    # cubature is another possibility but this is far slower!
    elif integration == "cubature": 
        contInteg, err = cubature(func2d, 2, 1, (rngx[0], rngy[0]), (rngx[1], rngy[1]))
        contInteg = float(contInteg)

    return contInteg

def numericalIntegrate(function, (rngx, rngy), SFprior=False):
    
    '''
    

    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    #compInteg = integrate.dblquad(function, rngx[0], rngx[1], rngy[0], rngy[1])
    Nx_int, Ny_int = 100, 250

    x_coords = np.linspace(rngx[0], rngx[1], Nx_int)
    y_coords = np.linspace(rngy[0], rngy[1], Ny_int)

    dx = ( rngx[1]-rngx[0] )/Nx_int
    dy = ( rngy[1]-rngy[0] )/Ny_int

    x_2d = np.tile(x_coords, ( len(y_coords), 1 ))
    y_2d = np.tile(y_coords, ( len(x_coords), 1 )).T
    z_2d = function((x_2d, y_2d))

    volume1 = ( (z_2d[:-1, :-1] + z_2d[1:, 1:])/2 ) * dx * dy
    volume2 = ( (z_2d[:-1, 1:] + z_2d[1:, :-1])/2 ) * dx * dy
    integral = ( np.sum(volume1.flatten()) + np.sum(volume2.flatten()) ) /2

    return integral

def simpsonIntegrate(function, (rngx, rngy)):

    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    #compInteg = integrate.dblquad(function, rngx[0], rngx[1], rngy[0], rngy[1])
    Nx_int, Ny_int = 100, 250

    x_coords = np.linspace(rngx[0], rngx[1], Nx_int)
    y_coords = np.linspace(rngy[0], rngy[1], Ny_int)

    dx = ( rngx[1]-rngx[0] )/Nx_int
    dy = ( rngy[1]-rngy[0] )/Ny_int

    x_2d = np.tile(x_coords, ( len(y_coords), 1 ))
    y_2d = np.tile(y_coords, ( len(x_coords), 1 )).T
    z_2d = function((x_2d, y_2d))
    
    z_intx = function((x_2d + dx/2, y_2d))[:-1, :]
    z_inty = function((x_2d, y_2d + dy/2))[:,:-1]                                      

    volume1 = ( (z_2d[:-1, :] + z_intx*4 + z_2d[1:, :] ) /6 ) * dx * dy
    volume2 = ( (z_2d[:, :-1] + z_inty*4 + z_2d[:, 1:] ) /6 ) * dx * dy
    integral = ( np.sum(volume1.flatten()) + np.sum(volume2.flatten()) ) /2

    return integral

'''

Functions for penalised grid fitting models

'''


def gridDistribution((nx, ny), (rngx, rngy), params):
    
    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    # nx, ny - number of x and y cells
    # rngx, rngy - ranges of x and y coordinates
    # params - nx x ny array of cell parameters
    
    boundsx = np.linspace(rngx[0], rngx[1], nx+1)
    boundsy = np.linspace(rngy[0], rngy[1], ny+1)

    #profile = interp.RegularGridInterpolator((boundsx, boundsy), params)
    #profile = interp.griddata((boundsx, boundsy), params, (boundsx, boundsy), method='cubic')
   
    inst = interp.interp2d(boundsy, boundsx, params, kind="cubic")
    #inst = interp.RectBivariateSpline(boundsx, boundsy, params)

    profile = lambda (a, b): np.diag(inst(a, b))

    return profile, inst

def gridIntegrate(function, (rngx, rngy)):
    
    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    #compInteg = integ.dblquad(function, rngx[0], rngx[1], rngy[0], rngy[1])
    compInteg, err = cubature(function, 2, 1, (rngx[0], rngy[0]), (rngx[1], rngy[1]))
    
    return compInteg

class PenalisedGridModel():
    
    '''


    Parameters
    ----------


    **kwargs
    --------


    Returns
    -------


    '''

    def __init__(self, xi, yi, Nside, rngx, rngy, zero_bounds = True):

        
        self.nx, self.ny = Nside
        self.rngx = rngx
        self.rngy = rngy

        # Starting values for parameters
        self.zero_bounds = zero_bounds
        if not zero_bounds:
        	self.params_i = np.zeros((self.ny+1, self.nx+1)) + 0.01
        else:
        	self.params_i = np.zeros((self.ny-1, self.nx-1)) + 0.01
        	self.grid = np.zeros((self.nx+1, self.ny+1))

        # Final optimal values for parameters
        self.params_f = []
        # Min value of parameters as prior
        #self.underPriors = np.zeros((self.ny+1, self.nx+1)) + 1e-10
        self.underPriors = 1e-10
        # Max value of parameters as prior
        #self.overPriors = np.zeros((self.ny+1, self.nx+1)) + 1e10
        self.overPriors = 1e10

        self.xi = xi
        self.yi = yi

        self.penWeight = 0.
        
        # Function which calculates the actual distribution
        self.distribution = gridDistribution
        
    def __call__(self, (xi, yi)):
        
        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        # Final interpolant is set after generateInterpolant is called
        
        return self.finalInterpolant((xi, yi))
        
    def generateInterpolant(self):
        
        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        """
        boundsx = np.linspace(self.rngx[0], self.rngx[1], self.nx+1)
        boundsy = np.linspace(self.rngy[0], self.rngy[1], self.ny+1)
        
        profile = interp.RegularGridInterpolator((boundsx, boundsy), self.params_f)#, kind='cubic')
        #profile = interp.griddata((boundsx, boundsy), params, (boundsx, boundsy), method='cubic')
		"""

        profile = self.distribution((self.nx,self.ny), (self.rngx, self.rngy), self.params_f)
        
        self.finalInterpolant = profile[0]

        self.instInterpolant = profile[1]

    def integral(self, params):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        dx = ( self.rngx[1]-self.rngx[0] )/self.nx
        dy = ( self.rngy[1]-self.rngy[0] )/self.ny
        volume1 = ( (params[:-1, :-1] + params[1:, 1:])/2 ) * dx * dy
        volume2 = ( (params[:-1, 1:] + params[1:, :-1])/2 ) * dx * dy
        integral = ( np.sum(volume1.flatten()) + np.sum(volume2.flatten()) ) /2

        """
		function = self.distribution((self.nx,self.ny), (self.rngx, self.rngy), params)
		y0 = lambda input: self.rngy[0]
		y1 = lambda input: self.rngy[1]

		integral = quad(function, self.rngx, self.rngy)
		#integral = integrate.dblquad(function, self.rngx[0], self.rngx[1], y0, y1)
		"""

        return integral

    def curvatureIntegral(self, params):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        dx = ( self.rngx[1]-self.rngx[0] )/self.nx
        dy = ( self.rngy[1]-self.rngy[0] )/self.ny
        ddiag = np.sqrt(dx**2 + dy**2)

        # Curvature in x=y direction
        c1 = ( params[2:, 2:] - 2*params[1:-1, 1:-1] + params[:-2, :-2] ) / (ddiag**2)

        # Curvature in x=-y direction
        c2 = ( params[2:, :-2] - 2*params[1:-1, 1:-1] + params[:-2, 2:] ) / (ddiag**2)

        # Curvature in y direction
        c3 = ( params[2:, :] - 2*params[1:-1, :] + params[:-2, :] ) / (dy**2)

        # Curvature in x direction
        c4 = ( params[:, 2:] - 2*params[:, 1:-1] + params[:, :-2] ) / (dx**2)

        integral = (np.sum( np.abs(c1.flatten()) ) +\
                    np.sum( np.abs(c2.flatten()) ) + \
                    np.sum( np.abs(c3.flatten()) ) + \
                    np.sum( np.abs(c4.flatten()) ) ) / 4

        return integral


    def optimizeParams(self):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        # nll is the negative lnlike distribution
        nll = lambda *args: -self.lnprob(*args)
        ll = lambda *args: self.lnprob(*args)

        # result is the set of theta parameters which optimise the likelihood given x, y, yerr
        result = op.minimize(nll, self.params_i.flatten(), method = 'Powell')
        #opt = Optimizers.nonMarkovOptimizer(ll, self.params_i.flatten())
        #result = opt()

        # Save evaluated parameters to internal values
        if not self.zero_bounds:
            self.params_f = result['x'].reshape((self.nx+1, self.ny+1))
        else:
            self.params_f = self.grid
            self.params_f[1:-1, 1:-1] = result['x'].reshape((self.nx-1, self.ny-1))
        
    # ln(Likelihood) based on a Poisson likelihood distribution
    def lnlike(self, params):
        
        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        # Reshape the parameters to make the next stage easy
        if not self.zero_bounds:
            param_set = params.reshape((self.nx+1, self.ny+1))
        else:
            param_set = self.grid
            param_set[1:-1, 1:-1] = params.reshape((self.nx-1, self.ny-1))

        model = self.distribution((self.nx, self.ny), (self.rngx, self.rngy), param_set)[0]
        contPoints = np.sum( np.log( model( (self.xi, self.yi) ) ) )
            
        # Integral over region for 2D Gaussian distribution
        contInteg = gridIntegrate(model, (self.rngx, self.rngy))
        #contInteg = self.integral(param_set)

        penWeight = self.penWeight#0#0.7
        curveInteg = self.curvatureIntegral(param_set)

        lnL = contPoints - contInteg - penWeight*curveInteg
        sys.stdout.write("\rlogL: %.2f, sum log(f(xi)): %.2f, integral: %.2f, curve pen: %.2f" % (lnL, contPoints, contInteg, curveInteg))
        sys.stdout.flush()

        return lnL

    # "uninformative prior" - uniform and non-zero within a specified range of parameter values
    def lnprior(self, params):
        
        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        param_set = params
        # Reshape the parameters to make the next stage easy
        
        prior = self.priorTest(param_set)
        
        if prior:
            return 0.0
        else: 
            return -np.inf

    # posterior probability function is proportional to the prior times the likelihood
    # lnpost = lnprior + lnlike
    def lnprob(self, params):

        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(params)
    
    # Returns a boolean true or false for whether all parameters lie within their range
    def priorTest(self, params):
        
        '''


        Parameters
        ----------


        **kwargs
        --------


        Returns
        -------


        '''

        Val = np.array(params).flatten()
        #minVal = np.array(self.underPriors).flatten()
        #maxVal = np.array(self.overPriors).flatten()
        minVal = self.underPriors
        maxVal = self.overPriors

        minBool = Val > minVal
        maxBool = Val < maxVal
        rngBool = minBool*maxBool
        
        solution = np.sum(rngBool) - len(Val)
        if solution == 0:
            return True
        else: return False