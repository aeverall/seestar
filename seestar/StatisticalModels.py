
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
from skopt import gp_minimize
import sys, os, time
from mpmath import *

# Import cubature for integrating over regions
from cubature import cubature


class GaussianEM():

    '''
    GaussianEM - Class for calculating bivariate Gaussian mixture model which best fits
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
        optimizeParams - Vary the parameters of the distribution using the given method to optimize the 
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
    
    def __init__(self, x=np.array(0), y=np.array(0), nComponents=0, rngx=(0,1), rngy=(0,1), runscaling=True):

        # Name of the model to used for reloading from dictionary
        self.modelname = self.__class__.__name__

        # Distribution from photometric survey for calculation of SF
        self.photoDF = None
        self.priorDF = False
        
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
        
        # Not run when loading class from dictionary
        if runscaling:
            # Real space parameters
            self.x = x
            self.y = y
            self.rngx, self.rngy = rngx, rngy
            # Statistics for feature scaling
            if len(x)>1:
                self.mux, self.sx = np.mean(x), np.std(x)
                self.muy, self.sy = np.mean(y), np.std(y)
            else:
                # SD=0 if only one point which causes problems!
                self.mux, self.sx = np.mean(x), (rngx[1]-rngx[0])/4
                self.muy, self.sy = np.mean(y), (rngy[1]-rngy[0])/4
            # Scaled parameters
            self.x_s, self.y_s = feature_scaling(x, y, self.mux, self.muy, self.sx, self.sy)
            self.rngx_s, self.rngy_s = feature_scaling(np.array(rngx), np.array(rngy), self.mux, self.muy, self.sx, self.sy)

        # Function which calculates the actual distribution
        self.distribution = bivGaussMix

        # Print out likelihood values as calculated
        self.runningL = True
        
    def __call__(self, x, y):

        '''
        __call__ - Returns the value of the smooth GMM distribution at the points x, y

        Parameters
        ----------
            x, y - float or np.array of floats
                - x and y coordinates of points at which to take the value of the GMM
                - From input - x is magnitude, y is colour

        Returns
        -------
            GMMval: float or np.array of floats
                - The value of the GMM at coordinates x, y
        '''

        # Scale x and y to correct region
        x, y = feature_scaling(x, y, self.mux, self.muy, self.sx, self.sy)
        rngx, rngy = feature_scaling(np.array(self.rngx), np.array(self.rngy), self.mux, self.muy, self.sx, self.sy)
        
        # Value of coordinates x, y in the Gaussian mixture model
        GMMval = self.distribution(self.params_f, x, y)

        if (type(GMMval) == np.array)|(type(GMMval) == np.ndarray)|(type(GMMval) == pd.Series): 
            # Not-nan input values
            notnan = (~np.isnan(x))&(~np.isnan(y))
            # Any values outside range - 0
            constraint = (x[notnan]>=rngx[0])&(x[notnan]<=rngx[1])&(y[notnan]>=rngy[0])&(y[notnan]<=rngy[1])
            GMMval[~notnan] = np.nan
            GMMval[notnan][~constraint] = 0.
        elif (type(GMMval) == float) | (type(GMMval) == np.float64):
            if (np.isnan(x))|(np.isnan(y)): GMMval = np.nan
            else: 
                constraint = (x>=rngx[0])&(x<=rngx[1])&(y>=rngy[0])&(y<=rngy[1])
                if not constraint:
                    GMMval = 0.
        else: raise TypeError('The type of the input variables is '+str(type(GMMval)))

        return GMMval

    def EM(self, params, prior=False): 

        '''
        EM - Expectation maximisation algorithm for Gaussian mixture model
            - Currently only gives correct result for photometric DF
            - Can't weight calculation of mu, sigma, w for a prior distribution (needed for SF)

        Parameters
        ----------
            params: array of floats - nx6 - where n is the number of components of the mixture
                - Initial/current parameters of Gaussian mixture model

        **kwargs
        --------
            prior=False: boolean
                - Is there a prior distribution to use (Such as the photoDF)
                - Don't actually know how to impose a prior yet

        Returns
        -------
            params: array of floats - nx6 - where n is the number of components of the mixture
                - New estimate of parameters of Gaussian mixture
        '''

        improvement = 100
        it = 0
        lnlike_old = self.lnlike(params)
        # Whilst improving by greater than 0.1%
        while improvement > 0.001:
            it+=1
            response = self.Expectation(params) 
            params = self.Maximisation(response, prior=prior)

            lnlike_new = self.lnlike(params)
            if 'lnlike_old' in locals():
                improvement = 100 * (lnlike_new - lnlike_old)/np.abs(lnlike_old)
            
            sys.stdout.write('\rold: %d, new: %d, improvement: %d' % (lnlike_old, lnlike_new, improvement))

            lnlike_old = lnlike_new

        print("\nIterations: %d" % it)
        return params

    def Expectation(self, params):

        '''
        Expectation - Expectation step of Expectation maximisation algorithm
            - Calculates the assosciation of each point with each component of the mixture (response)

        Parameters
        ----------
            params: array of floats - nx6 - where n is the number of components of the mixture
                - Initial/current parameters of Gaussian mixture model

        Inherited
        ---------
            x_s, y_s: array of float
                - Feature scaled x and y coordinates of points

        Returns
        -------
            response: array of floats - nxm
                - Assosciation of each point with each component of the mixture
                - n is the number of components
                - m is the number of points
        '''

        response = np.empty((params.shape[0],) + self.x_s.shape)
        for i in range(params.shape[0]):
            component = bivariateGauss(params[i,:], self.x_s, self.y_s)
            full = bivGaussMix(params, self.x_s, self.y_s)

            response[i,:] = component/full

        return response

    def Maximisation(self, response, prior=False):

        '''
        Maximisation - Maximisation step of Expectation maximisation algorithm
            - Calculates the parameters of mixture components using the points weighted by
            their assosciations with each component.

        Parameters
        ----------
            response: array of floats - nxm
                - Assosciation of each point with each component of the mixture
                - n is the number of components
                - m is the number of points

        kwargs
        ------
            prior: bool
                - Whether to use a prior distribution function when calculating parameters
                - Not yet sure how this is done

        Inherited
        ---------
            x_s, y_s: array of float
                - Feature scaled x and y coordinates of points

        Returns
        -------
            params: array of floats - nx6 - where n is the number of components of the mixture
                - Improved estimate for parameters of Gaussian mixture model

        '''

        shape = self.x_s.shape
        X = np.vstack((self.x_s.ravel(), self.y_s.ravel())).T

        params = np.empty(self.params_i.shape)
        for i in range(response.shape[0]):

            MU = np.dot(response[i,:], X) / np.sum(response[i,:]) 

            diff = X-MU
            XX = np.array([np.outer(elem, elem) for elem in diff])
            # Weight matrices by response
            XX = (response[i,:]*XX.T).T
            sigma = np.sum(XX, axis=0) / np.sum(response[i,:])
            determinant = np.linalg.det(sigma)

            weight = np.sum(response[i,:])
            if prior: weight /= len(self.x_s.ravel())

            params[i,:] = np.array((MU[0], sigma[0,0], MU[1], sigma[1,1], weight, sigma[1,0]))

        return params
        
    def optimizeParams(self, method = "Powell"):

        '''
        optimizeParams - Initialise and optimize parameters of Gaussian mixture model.

        **kwargs
        --------
            method = "Powell" - str
                - The scipy.optimize.minimize method used to vary the parameters of the distribution
                in order to find the minimum negative log likelihooh (min ( -log(L) ))

        Returns
        -------
            result - dict
                - Output of scipy.optimize.minimise showing the details of the process
        '''

        # Set initial parameters
        finite = False
        a = 0
        while not finite:
            self.params_i, self.params_l, self.params_u = initParams(self.nComponents, self.rngx_s, self.rngy_s, self.priorDF, nstars=len(self.x_s))
            self.param_shape = self.params_i.shape
            self.s_min = self.params_l[0,1] # same as sxx lower bound
            lnp = self.lnprob(self.params_i)
            finite = np.isfinite(lnp)
            a+=1
            if a==100: 
                print("...failed to initialise params on field")
                finite=True
            if self.runningL:
                sys.stdout.write("\r"+str(self.params_i))
                if finite: print("")


        params = self.params_i

        # Test runs different versions of the optimizer to find the best.
        test=False
        if test:
            start = time.time()
            bounds = list(zip(self.params_l.ravel(), self.params_u.ravel()))
            # Run scipy optimizer
            resultSLSQP = self.optimize(params, "SLSQP", bounds)
            paramsSLSQP = resultSLSQP["x"].reshape(self.param_shape)
            # Check likelihood for parameters
            lnlikeSLSQP = self.lnlike(paramsSLSQP)
            print("\n %s: lnprob=%d, time=%d" % ("SLSQP", self.lnprob(paramsSLSQP), time.time()-start))

            start = time.time()
            bounds=None
            # Run expectation max
            paramsEM = self.EM(params, prior=self.priorDF)
            # Run scipy optimize
            resultEM = self.optimize(paramsEM, method, bounds)
            paramsEM = resultEM["x"].reshape(self.param_shape)
            # Check likelihood for parameters
            lnlikeEM = self.lnlike(paramsEM)
            print("\n EM + %s: lnprob=%d, time=%d" % (method, self.lnprob(paramsEM), time.time()-start))

            start = time.time()
            bounds=None
            # Run scipy optimizer
            resultOP = self.optimize(params, method, bounds)
            paramsOP = resultOP["x"].reshape(self.param_shape)
            # Check likelihood for parameters
            lnlikeOP = self.lnlike(paramsOP)
            print("\n %s: lnprob=%d, time=%d" % (method, self.lnprob(paramsOP), time.time()-start))

            if lnlikeOP>lnlikeEM: result = resultOP
            else: result = resultEM

        else:
            start = time.time()
            bounds = None
            # Run scipy optimizer
            if self.runningL: print("\nInitparam likelihood: %d" % self.lnprob(params))
            resultOP = self.optimize(params, method, bounds)
            paramsOP = resultOP["x"].reshape(self.param_shape)
            # Check likelihood for parameters
            lnlikeOP = self.lnlike(paramsOP)
            if self.runningL: print("\n %s: lnprob=%d, time=%d" % (method, self.lnprob(paramsOP), time.time()-start))
            result = resultOP

        # Save evaluated parameters to internal values
        self.params_f = result["x"].reshape(self.param_shape)
       
        return result

    def optimize(self, params, method, bounds):

        '''
        optimize - Run scipy.optimize.minimize to determine the optimal parameters for the distribution

        Parameters
        ----------
            params: array of float
                - Initial parameters of the distribution

            method: str
                - Method to be used by optimizer, e.g. "Powell"

            bounds: list of tup of float or None
                - Boundaries of optimizing region
                - Only for optimization methods which take bounds

        Returns
        -------
            result: dict
                - Output of scipy.optimize
        '''

        # To clean up any warnings from optimize
        invalid = np.seterr()['invalid']
        divide = np.seterr()['divide']
        over = np.seterr()['over']
        np.seterr(invalid='ignore', divide='ignore', over='ignore')
        # result is the set of theta parameters which optimize the likelihood given x, y, yerr
        result = op.minimize(self.nll, params.ravel(), method=method, bounds=bounds)

        # Potential to use scikit optimize
        #bounds = list(zip(self.params_l.ravel(), self.params_u.ravel()))
        #result = gp_minimize(self.nll, bounds)

        # To clean up any warnings from optimize
        np.seterr(invalid=invalid, divide=divide, over=over)
        if self.runningL: print("")

        return result

    def nll(self, *args):

        """
        nll - Negative log likelihood for use in the optimizer
        """
        lp = -self.lnprob(*args)
        return lp

    def lnprob(self, params):

        '''
        lnprob - ln of the posterior probability of the distribution given the parameters.
               - posterior probability function is proportional to the prior times the likelihood
               - lnpost = lnprior + lnlike

        Parameters
        ----------
            params: array of float
                - Initial parameters of the distribution

        Returns
        -------
            lnprior(params)+lnlike(params) - float
                - ln of the posterior probability
                - -np.inf if the prior is false and hence prob=0 - lnprob=-inf
        '''

        # Reshape parameters for testing
        params = params.reshape(self.param_shape)

        lp = self.lnprior(params)

        # if prior not satisfied, don't calculate lnlike
        if not np.isfinite(lp): return -np.inf
        # if prior satisfied
        else: return lp + self.lnlike(params) 

    def lnlike(self, params):

        '''
        lnlike - The poisson likelihood disrtibution of the Gaussian mixture model given the observed points

        Parameters
        ----------
            params: array of float
                - Initial parameters of the distribution

        Returns
        -------
            contPoints-contInteg - float
                - lnL for the GMM parameters given the observed points
        '''

        # If the DF has already been calculated, directly optimize the SF
        if self.priorDF: function = lambda a, b: self.photoDF(*(a,b)) * self.distribution(params, a, b)
        else: function = lambda a, b: self.distribution(params, a, b)
        
        # Point component of poisson log likelihood: contPoints \sum(\lambda(x_i))
        model = function(*(self.x_s, self.y_s))
        contPoints = np.sum( np.log(model) )
        # Integral of the smooth function over the entire region
        contInteg = integrationRoutine(function, params, self.nComponents, *(self.rngx_s, self.rngy_s))

        lnL = contPoints - contInteg
        if self.runningL:
            sys.stdout.write("\rlogL: %.2f, sum log(f(xi)): %.2f, integral: %.2f            " % (lnL, contPoints, contInteg))
            sys.stdout.flush()

        return contPoints - contInteg

    def lnprior(self, params):
        
        '''
        lnprior - The test of the parameters of the Gaussian Mixture Model against the prior

        Parameters
        ----------
            params: array of float
                - Initial parameters of the distribution

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
        if self.priorDF:
            function = lambda a, b: self.distribution(params, a, b)
            prior = prior & SFprior(function, *(self.rngx_s, self.rngy_s))
            if not prior: return -np.inf

        # All prior tests satiscied
        return 0.0

    def prior_bounds(self, params):
        
        '''
        prior_bounds - Testing the parameters of the GMM against the upper and lower limits
            - uninformative prior - uniform and non-zero within a specified range of parameter values

        Parameters
        ----------
            params - array of floats
                - Values of parameters in the Gaussian Mixture model

        Inherited
        ---------
            params_l, params_u: array of float
                - Lower and upper bounds of parameters

        Returns
        -------
             - prior: bool
                - True if parameters satisfy constraints. False if not.
        '''

        # Total is 0 if all parameters within priors
        total = np.sum(params <= self.params_l) + np.sum(params >= self.params_u)
        # prior True if all parameters within priors
        prior = total == 0
        
        return prior

    def sigma_bound(self, params):
        
        '''
        sigma_bound - Priors placed on the covariance matrix in the parameters

        Parameters
        ----------
            params - array of floats
                - Values of parameters in the Gaussian Mixture model

        Returns
        -------
            - bool
            - True if good covariance matrix, False if not
            - Good covariance has det>0
        '''

        for i in range(params.shape[0]):
            sigma = np.array([[params[i,1], params[i,5]], [params[i,5], params[i,3]]])
            try: 
                eigvals = np.linalg.eigvals(sigma)
                det = np.linalg.det(sigma)
            except np.linalg.LinAlgError:
                print(params)
                print(sigma)
                raise ValueError('bad sigma')
                
            # Eigenvalues in allowed region
            result = np.sum(eigvals < self.s_min)
            if result != 0: return False
            
            # Non-negative determinant
            if det<0: return False
            
        return True


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

        function = lambda a, b: self.distribution(self.params_f, a, b)
        cub_func = lambda X: self.distribution(self.params_f, X[0], X[1])

        real_val, err = cubature(function, 2, 1, (self.rngx_s[0], self.rngy_s[0]), (self.rngx_s[1], self.rngy_s[1]))
        calc_val = integrationRoutine(function, self.params_f, self.nComponents, *(self.rngx_s, self.rngy_s), integration=integration)

        percent = ((calc_val - float(real_val))/calc_val)*100
        cubature_percent = 100*float(err)/float(real_val)

        print("\nThe error in the linear numerical integral was %.3E%%" % float(percent))
        print("\nThe cubature calculation error is quoted as %.3E or %.3E%%" % (float(err), cubature_percent)  )

        return calc_val, real_val, err

    def stats(self):
        
        '''
        stats - Prints out the statistics of the model fitting.

        Such as parameters, likelihoods, integration calculations

        Inherited
        ---------
            params_f: arr of float
                - Final parameter values after optimization

            rngx_s, rngy_s: tuple of float
                - Feature scaled x and y range

        Returns
        -------
            None
        '''

        print("Parameters:")
        for i in range(self.params_f.shape[0]):
            print("Parameters for component %d:" % i)
            mu = np.array((self.params_f[i,0], self.params_f[i,2]))
            sigma = np.array([[self.params_f[i,1], self.params_f[i,5]], [self.params_f[i,5], self.params_f[i,3]]])

            X = np.vstack((self.x_s.ravel(), self.y_s.ravel())).T
            p = self.params_f[i,4]*bivariateGaussVector(X, mu, sigma)

            print("mu x and y: {}, {}".format(*mu))
            print("covariance matrix: {}".format(str(sigma)))
            print("Weight: {}".format(self.params_f[i,4]))
            print("Contribution: {}.\n".format(p))


        nll = -self.lnprob(self.params_f)
        print("Negative log likelihood: {:.3f}".format(nll))
        lnlike = self.lnlike(self.params_f)
        print("Log likelihood: {:.3f}".format(lnlike))
        lnprior = self.lnprior(self.params_f)
        print("Log prior: {:.3f}".format(lnprior))
        #int_calc, int_real, err = self.testIntegral()
        #print("Integration: Used - {:.3f}, Correct - {:.3f} (Correct undertainty: {:.2f})".format(int_calc, int_real, err))
        function = lambda a, b: self.distribution(self.params_f, a, b)
        calc_val = numericalIntegrate(function, *(self.rngx_s, self.rngy_s))
        calc_val2 = numericalIntegrate(function, *(self.rngx_s, self.rngy_s), Nx_int=500, Ny_int=500)
        print("Integration: Used {:.3f}, Half spacing {:.3f}".format(calc_val, calc_val2))


def initParams(nComponents, rngx, rngy, priorDF, nstars=1):

    '''
    initParams - Specify the initial parameters for the Gaussian mixture model as well as 
                the lower and upper bounds on parameters (used as prior values in the optimization)

    Parameters
    ----------
        nComponents: int
            - Number of components of the Gaussian Mixture model

        rngx, rngy: tuple of float
            - Range of region in x and y axis

        priorDF: bool
            - Is there a prior distribution function?
            - True if calculating for selection function

    Returns
    -------
        params_i - array of floats
            - initial parameters for the Gaussian mixture model
        params_l - array of floats
            - lower bounds on the values of GMM parameters
        params_u - array of floats
            - upper bounds on the values of GMM parameters
    '''

    # Initial guess parameters for a bivariate Gaussian
    # Randomly initialise mean between -1 and 1
    mux_i, muy_i = np.random.rand(2, nComponents)*2-1
    # SD 1. as that's what feature scaling has set
    sxx_i, syy_i = 1., 1.
    # 0. covariance
    sxy_i = 0.
    # Weights sum to 1
    if priorDF: w_i = .1 / nComponents
    else: w_i = nstars/ nComponents

    # Lower and upper bounds on parameters
    # Mean at edge of range
    mux_l, mux_u = rngx
    muy_l, muy_u = rngy
    # Zero standard deviation to inf
    sxx_l, syy_l = 0, 0
    sxx_u, syy_u = np.inf, np.inf #rngx[1]-rngx[0], rngy[1]-rngy[0]
    # Covariance must be in range -1., 1.
    sxy_l, sxy_u = -np.inf, np.inf
    # Zero weight
    w_l = 0.
    w_u = np.inf

    params_i = np.empty((nComponents, 6))
    params_l = np.empty((nComponents, 6))
    params_u = np.empty((nComponents, 6))
    for i in range(nComponents):
        params_i[i,:] = np.array((mux_i[i], sxx_i, muy_i[i], syy_i, w_i, sxy_i))
        params_l[i,:] = np.array((mux_l, sxx_l, muy_l, syy_l, w_l, sxy_l))
        params_u[i,:] = np.array((mux_u, sxx_u, muy_u, syy_u, w_u, sxy_u))

    return params_i, params_l, params_u


def SFprior(function, rngx, rngy):

    '''
    SFprior - The selection function has to be between 0 and 1 everywhere.
        - informative prior

    Parameters
    ----------
        function - function
            - The selection function interpolant over the region, R.

        rngx, rngy: tuple of float
            - Range of region in x and y axis

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

    SF = function(*(x_2d, y_2d))

    prior = not np.max(SF)>1

    return prior

def bivariateGauss(params, x, y):

    '''
    bivariateGauss - Calculation of bivariate Gaussian distribution.

    Parameters
    ----------
        params - arr of float - length 6 
            - Parameters of the bivariate gaussian
        x, y - arr of float
            - x and y coordinates of points being tested
    Returns
    -------
        p - arr of float
            - bivariate Gaussian value for each point in x, y
    '''

    shape = x.shape
    X = np.vstack((x.ravel(), y.ravel())).T

    mu = np.array((params[0], params[2]))
    sigma = np.array([[params[1], params[5]], [params[5], params[3]]])
    weight= params[4]

    # Inverse covariance
    inv_cov = np.linalg.inv(sigma)
    # Separation of X from mean
    X = X-mu
    # X^T * Sigma
    X_cov = np.dot(X, inv_cov)
    # X * Sigma * X
    X_cov_X = np.sum(X_cov*X, axis=1)
    # Exponential
    e = np.exp(-X_cov_X/2)

    # Normalisation term
    det_cov = np.linalg.det(sigma)
    norm = 1/np.sqrt( ((2*np.pi)**2) * det_cov)

    p = weight*norm*e
    return p.reshape(shape)

def bivGaussMix(params, x, y):

    '''
    bivariateGauss - Calculation of bivariate Gaussian mixture distribution.

    Parameters
    ----------
        params - arr of float - nx6
            - Parameters of the bivariate gaussian
        x, y - arr of float
            - x and y coordinates of points being tested
    Returns
    -------
        p - arr of float
            - bivariate Gaussian mixture value for each point in x, y
    '''
    
    p = 0
    for i in range(params.shape[0]):
        p += bivariateGauss(params[i,:], x, y)

    return p

def feature_scaling(x, y, mux, muy, sx, sy):

    '''
    feature_scaling - Scales features to a zero mean and unit standard deviation

    Parameters
    ----------
        x, y - arr of float
            - x and y coordinates of points
        mux, muy - float
            - Mean of distribution in x and y coordinates
        sx, sy - floats
            - Standard deviation of coordinates in x and y coordinates
    Returns
    -------
        scalex, scaley - arr of float
            - x and y coordinates scaled by feature scaling

    '''

    scalex = (x-mux)/sx
    scaley = (y-muy)/sy

    return scalex, scaley


"""

INTEGRATION ROUTINES

"""

def integrationRoutine(function, param_set, nComponents, rngx, rngy, integration = "trapezium"):

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

        rngx, rngy - tuple of floats
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
    elif integration == "trapezium": contInteg = numericalIntegrate(function, *(rngx, rngy))
    # simpson is a quadratic approximation to the integral - reasonably fast - ~1% accurate
    elif integration == "simpson": contInteg = simpsonIntegrate(function, *(rngx, rngy))
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

def numericalIntegrate(function, rngx, rngy, Nx_int=250, Ny_int=250):
    
    '''
    numericalIntegrate - Integrate over region using the trapezium rule

    Parameters
    ----------
        function - function or interpolant
            - The function to be integrated over the specified region of space

        nComponents - int
            - Number of components of the GMM.

        rngx, rngy - tuple of floats
            - Boundary of region of colour-magnitude space being calculated.
        

    **kwargs
    --------
        Nx_int, Ny_int: int
            - Number of grid spacings to place along the x and y axes

    Returns
    -------
        integral: float
            - Integral over the region
    '''

    #compInteg = integrate.dblquad(function, rngx[0], rngx[1], rngy[0], rngy[1])
    Nx_int, Ny_int = (Nx_int, Ny_int)

    x_coords = np.linspace(rngx[0], rngx[1], Nx_int)
    y_coords = np.linspace(rngy[0], rngy[1], Ny_int)

    dx = ( rngx[1]-rngx[0] )/Nx_int
    dy = ( rngy[1]-rngy[0] )/Ny_int

    x_2d = np.tile(x_coords, ( len(y_coords), 1 ))
    y_2d = np.tile(y_coords, ( len(x_coords), 1 )).T
    z_2d = function(*(x_2d, y_2d))

    volume1 = ( (z_2d[:-1, :-1] + z_2d[1:, 1:])/2 ) * dx * dy
    volume2 = ( (z_2d[:-1, 1:] + z_2d[1:, :-1])/2 ) * dx * dy
    integral = ( np.sum(volume1.flatten()) + np.sum(volume2.flatten()) ) /2

    return integral

def simpsonIntegrate(function, rngx, rngy):

    '''
    simpsonIntegrate - Integrate over region using simson's rule

    Parameters
    ----------
        function - function or interpolant
            - The function to be integrated over the specified region of space

        nComponents - int
            - Number of components of the GMM.

    Returns
    -------
        integral: float
            - Integral over the region
    '''

    #compInteg = integrate.dblquad(function, rngx[0], rngx[1], rngy[0], rngy[1])
    Nx_int, Ny_int = 100, 250

    x_coords = np.linspace(rngx[0], rngx[1], Nx_int)
    y_coords = np.linspace(rngy[0], rngy[1], Ny_int)

    dx = ( rngx[1]-rngx[0] )/Nx_int
    dy = ( rngy[1]-rngy[0] )/Ny_int

    x_2d = np.tile(x_coords, ( len(y_coords), 1 ))
    y_2d = np.tile(y_coords, ( len(x_coords), 1 )).T
    z_2d = function(*(x_2d, y_2d))
    
    z_intx = function(*(x_2d + dx/2, y_2d))[:-1, :]
    z_inty = function(*(x_2d, y_2d + dy/2))[:,:-1]                                      

    volume1 = ( (z_2d[:-1, :] + z_intx*4 + z_2d[1:, :] ) /6 ) * dx * dy
    volume2 = ( (z_2d[:, :-1] + z_inty*4 + z_2d[:, 1:] ) /6 ) * dx * dy
    integral = ( np.sum(volume1.flatten()) + np.sum(volume2.flatten()) ) /2

    return integral

def gridIntegrate(function, rngx, rngy):
    
    '''
    gridIntegrate - Integrate over the grid when using PenalisedGridModel

    Parameters
    ----------
        function - function or interpolant
            - The function to be integrated over the specified region of space

        nComponents - int
            - Number of components of the GMM.

    Returns
    -------
        compInteg: float
            - Integral over the region
    '''

    #compInteg = integ.dblquad(function, rngx[0], rngx[1], rngy[0], rngy[1])
    compInteg, err = cubature(function, 2, 1, (rngx[0], rngy[0]), (rngx[1], rngy[1]))
    
    return compInteg

'''

Functions for penalised grid fitting models

'''

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


        # Name of the model to used for reloading from dictionary
        self.modelname = self.__class__.__name__
        
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
        
    def __call__(self,xi, yi):
        
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

        # result is the set of theta parameters which optimize the likelihood given x, y, yerr
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

def gridDistribution(nx, ny, rngx, rngy, params):
    
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

    profile = lambda a, b: np.diag(inst(a, b))

    return profile, inst

"""



"""


# Used when Process == "Number"
class FlatRegion:

    '''
    FlatRegion - Model with constant value over entire region

    Parameters
    ----------
        value: float
            - value of selection function in region
        rangex, rangey: tuple of float
            - x and y ranges of region

    Returns
    -------
        result: float or arr of float
            - Value of selection function at x, y coordinates
    '''

    def __init__(self, value, rangex, rangey):

        # Name of the model to used for reloading from dictionary
        self.modelname = self.__class__.__name__

        self.value = value
        self.rangex = rangex
        self.rangey = rangey

    def __call__(self, x, y):

        '''
        __call__ - Calculate the selection function at given coordinates

        Parameters
        ----------
            x, y: arr of float
                - Coordinates at which we are calculating the selecion function

        Inherited
        ---------
            rangex, rangey: tuple of float
                - x and y ranges of region

        Returns
        -------
            result: float or arr of float
                - Value of selection function at x, y coordinates

        '''

        result = np.zeros(np.shape(x))
        result[(x>self.rangex[0]) & \
                (x<self.rangex[1]) & \
                (y>self.rangey[0]) & \
                (y<self.rangey[1])] = self.value

        return result

"""

Functions which I don't think are used any more

"""


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


class Empty():

    '''
    Empty - For fields with no stars present

    Functions
    ---------
    __call__
    '''
    
    def __init__(self, runscaling=True):

        # Name of the model to used for reloading from dictionary
        self.modelname = self.__class__.__name__

    def __call__(self, x, y):

        return np.zeros(np.shape(x))