
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
        self.params_i = None
        # Prior limits on parameters
        self.params_l, self.params_u = None, None
        # Final optimal values for parameters
        self.params_f = None
        # Shape of parameter set (number of components x parameters per component)
        self.param_shape = ()
        
        # Real space parameters
        self.x = x
        self.y = y
        self.rngx, self.rngy = rngx, rngy

        # Scaled parameters
        self.f_scale = feature_scaling(x, y)
        self.x_s, self.y_s = self.f_scale(x, y)
        self.rngx_s, self.rngy_s = self.f_scale(np.array(rngx), np.array(rngy))

        # Function which calculates the actual distribution
        self.distribution = multiDistributionVector

        # Print out likelihood values as calculated
        self.runningL = True
        
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

        # Scale x and y to correct region
        x, y = self.f_scale(x, y)
        
        # Value of coordinates x, y in the Gaussian mixture model
        GMMval = self.distribution(self.params_f, x, y)

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
        #self.params_i, self.params_l, self.params_u = self.initParams()
        self.params_i, self.params_l, self.params_u = initParams(self.nComponents, self.rngx_s, self.rngy_s, self.priorDFbool)

        self.param_shape = self.params_i.shape
        self.s_min = 0.1

        # nll is the negative lnlike distribution
        nll = lambda *args: -self.lnprob(*args)
        # result is the set of theta parameters which optimise the likelihood given x, y, yerr
        result = op.minimize(self.nll, self.params_i.ravel(), method = method, bounds=zip(self.params_l.ravel(), self.params_u.ravel()))

        # Save evaluated parameters to internal values
        self.params_f = result["x"].reshape(self.param_shape)
       
        return result

    def nll(self, *args):

        lp = -self.lnprob(*args)

        if self.runningL:
            sys.stdout.write("\rlp: %.2f" % (lp))
            sys.stdout.flush()

        return lp

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

        # Reshape parameters for testing
        params = params.reshape(self.param_shape)

        if np.sum(np.isnan(params))>0:
            print('fu*k')

        lp = self.lnprior(params)

        # if prior not satisfied, don't calculate lnlike
        if not np.isfinite(lp): return -np.inf
        # if prior satisfied
        else: return lp + self.lnlike(params) 

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

        # If the DF has already been calculated, directly optimise the SF
        if self.priorDFbool: function = lambda (a, b): self.photoDF((a,b)) * self.distribution(params, a, b)
        else: function = lambda (a, b): self.distribution(params, a, b)

        # Point component of poisson log likelihood: contPoints \sum(\lambda(x_i))
        model = function((self.x_s, self.y_s))
        contPoints = np.sum( np.log(model) )
        # Integral of the smooth function over the entire region
        contInteg = integrationRoutine(function, params, self.nComponents, (self.rngx_s, self.rngy_s))
        lnL = contPoints - contInteg

        #if self.runningL:
        #    sys.stdout.write("\rlogL: %.2f, sum log(f(xi)): %.2f, integral: %.2f            " % (lnL, contPoints, contInteg))
        #    sys.stdout.flush()
            
        #if np.isnan(lnL): 
        #    print("\n"+str(params))
        #    print(function((self.x_s, self.y_s)))

        return contPoints - contInteg

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

        # Test parameters against boundary values
        prior = self.prior_bounds(params)
        if not prior: return -np.inf

        # Test parameters against variance
        if prior: prior = prior & self.sigma_bound(params)
        if not prior: return -np.inf

        # Prior on spectro distribution that it must be less than the photometric distribution
        if self.priorDFbool:
            function = lambda (a, b): self.distribution(params, a, b)
            prior = prior & SFprior(function, (self.rngx_s, self.rngy_s))
            if not prior: return -np.inf

        # All prior tests satiscied
        return 0.0

    def prior_bounds(self, params):
        
        '''
        prior_bounds - Testing the parameters of the GMM against the upper and lower limits specified in
                    self.initParams
            - uninformative prior - uniform and non-zero within a specified range of parameter values

        Parameters
        ----------
            params - array of floats
                - Values of parameters in the Gaussian Mixture model

        Returns
        -------
             - bool
                - True if parameters satisfy constraints. False if not.
        '''

        # Total is 0 if all parameters within priors
        total = np.sum(params <= self.params_l) + np.sum(params >= self.params_u)
        # prior True if all parameters within priors
        prior = total == 0
        
        return prior

    def sigma_bound(self, params):

        '''
        varTest - tests whether variance is constrained within prior limits

        Parameters
        ----------
            params - array of floats
                - Values of parameters in the Gaussian Mixture model
        Returns
        -------
             - bool
                - True if parameters satisfy constraints. False if not.
        '''

        for i in range(params.shape[0]):
            sigma = np.array([[params[i,1], params[i,5]], [params[i,5], params[i,3]]])
            try: eigvals = np.linalg.eigvals(sigma)
            except np.linalg.LinAlgError:
                print(params)
                print(sigma)
                raise ValueError('bad sigma')
            result = np.sum(eigvals < self.s_min)

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

        function = lambda (a, b): self.distribution(self.params_f, a, b)

        real_val, err = cubature(function, 2, 1, (self.rngx_s[0], self.rngy_s[0]), (self.rngx_s[1], self.rngy_s[1]))
        calc_val = integrationRoutine(function, self.params_f, self.nComponents, (self.rngx_s, self.rngy_s), integration=integration)

        percent = ((calc_val - float(real_val))/calc_val)*100
        cubature_percent = 100*float(err)/float(real_val)

        print("\nThe error in the linear numerical integral was %.3E%%" % float(percent))
        print("\nThe cubature calculation error is quoted as %.3E or %.3E%%" % (float(err), cubature_percent)  )

        return calc_val, real_val, err


def initParams(nComponents, rngx, rngy, priorDF):

    '''
    initParams - Specify the initial parameters for the Gaussian mixture model as well as 
                the lower and upper bounds on parameters (used as prior values in the optimization)

    Returns
    -------
        parameters_i - array of floats
            - initial parameters for the Gaussian mixture model
        parameters_u - array of floats
            - "under parameters" - lower limits on the values of GMM parameters
        parameters_o - array of floats
            - "over parameters" - upper limits on the values of GMM parameters
    '''

    # Initial guess parameters for a bivariate Gaussian
    # Mean in middle of range
    mux_i, muy_i = 0, 0
    # SD as 1/10 of range
    sxx_i, syy_i = 1., 1.
    # 0. covariance
    sxy_i = 0.
    # SF with DF prior - use 0.1/ncomponent start (can only really max to 1)
    if priorDF: A_i =  0.1 / nComponents 
    # DF component - can go up to the number of photometric stars
    else: A_i = 1.

    # Lower and upper bounds on parameters
    # Mean at edge of range
    mux_l, mux_u = rngx
    muy_l, muy_u = rngy
    # Zero standard deviation to the size of the region
    sxx_l, syy_l = 0, 0
    sxx_u, syy_u = rngx[1]-rngx[0], rngy[1]-rngy[0]
    # Covariance must be in range -1., 1.
    sxy_l, sxy_u = -1., 1.
    # Zero amplitude
    A_l = 0.
    # If calculating SF, A_o cannot be larger than 1
    if priorDF: A_u = 1.
    else: A_u = np.inf

    params_i = np.array((mux_i, sxx_i, muy_i, syy_i, A_i, sxy_i))
    params_l = np.array((mux_l, sxx_l, muy_l, syy_l, A_l, sxy_l))
    params_u = np.array((mux_u, sxx_u, muy_u, syy_u, A_u, sxy_u))

    # Initial parameters for a Double bivariate Gaussian
    params_i = np.tile(params_i, (nComponents, 1))
    params_l = np.tile(params_l, (nComponents, 1))
    params_u = np.tile(params_u, (nComponents, 1))

    return params_i, params_l, params_u


def feature_scaling(x, y):

    mux = np.mean(x)
    sx = np.std(x)

    muy = np.mean(y)
    sy = np.std(y)

    scale = lambda x, y: ((x-mux)/sx, (y-muy)/sy)

    return scale


def SFprior(function, (rngx, rngy)):

    '''
    SFprior - The selection function has to be between 0 and 1 everywhere.
        - informative prior
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

def bivariateGaussVector(x, mu, covariance):

    '''
    bivariateGaussVector

    Parameters
    ----------

    Returns
    -------

    '''

    # Inverse covariance
    inv_cov = np.linalg.inv(covariance)
    # Separation of X from mean
    X = x-mu
    # X^T * Sigma
    X_cov = np.dot(X, inv_cov)
    # X * Sigma * X
    X_cov_X = np.sum(X_cov*X, axis=1)
    # Exponential
    e = np.exp(-X_cov_X/2)

    # Normalisation term
    det_cov = np.linalg.det(covariance)
    norm = 1/np.sqrt( ((2*np.pi)**2) * det_cov)

    return norm*e

def multiDistributionVector(params, x, y):

    '''
    multiDistribution 

    Parameters
    ----------
        params - list of floats
            - Values of parameters for the Gaussian.

        x, y - float or np.array of floats
            - Coordinates at which the GMM is calculated.

    Returns
    -------
        p - Value of the GMM at coordinates x, y
    '''
    shape = x.shape
    X = np.vstack((x.ravel(), y.ravel())).T
    
    p = 0
    for i in range(params.shape[0]):
        mu = np.array((params[i,0], params[i,2]))
        sigma = np.array([[params[i,1], params[i,5]], [params[i,5], params[i,3]]])
        p += params[i,4]*bivariateGaussVector(X, mu, sigma)

    return p.reshape(shape)
    
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


"""

INTEGRATION ROUTINES

"""

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