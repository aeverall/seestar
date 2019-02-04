
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
import scipy.special as spec
#from skopt import gp_minimize
import sys, os, time
from mpmath import *

# Import cubature for integrating over regions
#from cubature import cubature


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

    def __init__(self, x=np.array(0), y=np.array(0), sig_xy=None,
                nComponents=0, rngx=(0,1), rngy=(0,1), runscaling=True, runningL=True, s_min=0.1):

        # Iteration number to update
        self.iter_count = 0

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

        # Boundary on minimum std
        self.s_min=s_min

        # Coordinate covariance matrix
        if sig_xy is None:
            z_ = np.zeros(len(x))
            sig_xy = np.array([[z_, z_],[z_, z_]]).transpose(2,0,1)
        self.sig_xy = sig_xy

        self.runscaling = runscaling
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
            self.sig_xy_s = covariance_scaling(self.sig_xy, self.sx, self.sy)
        else:
            self.x_s, self.y_s = x, y
            self.rngx_s, self.rngy_s = rngx, rngy
            self.sig_xy_s = sig_xy

        # Function which calculates the actual distribution
        self.distribution = bivGaussMix_vect_revamp

        # Print out likelihood values as calculated
        self.runningL = runningL

        Nx_int, Ny_int = (100, 100)
        x_coords = np.linspace(self.rngx_s[0], self.rngx_s[1], Nx_int)
        y_coords = np.linspace(self.rngy_s[0], self.rngy_s[1], Ny_int)
        self.x_2d, self.y_2d = np.meshgrid(x_coords, y_coords)

    def __call__(self, x, y, components=None):

        '''
        __call__ - Returns the value of the smooth GMM distribution at the points x, y

        Parameters
        ----------
            x, y - float or np.array of floats
                - x and y coordinates of points at which to take the value of the GMM
                - From input - x is magnitude, y is colour

            components=None:
                - List of components to check for distribution values

        Returns
        -------
            GMMval: float or np.array of floats
                - The value of the GMM at coordinates x, y
        '''

        # Scale x and y to correct region - Currently done to params_f - line 371  - but could change here instead
        #x, y = feature_scaling(x, y, self.mux, self.muy, self.sx, self.sy)
        #rngx, rngy = feature_scaling(np.array(self.rngx), np.array(self.rngy), self.mux, self.muy, self.sx, self.sy)
        rngx, rngy = np.array(self.rngx), np.array(self.rngy)

        # Value of coordinates x, y in the Gaussian mixture model
        if components is None: components = np.arange(self.nComponents)
        GMMval = self.distribution(self.params_f[components, :], x, y)

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

    def optimizeParams(self, method="Powell"):

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
            self.params_i, self.params_l, self.params_u = \
                initParams_revamp(self.nComponents, self.rngx_s, self.rngy_s, self.priorDF,
                                nstars=len(self.x_s), runscaling=self.runscaling, l_min=self.s_min)
            self.param_shape = self.params_i.shape
            lnp = self.lnprob(self.params_i)
            finite = np.isfinite(lnp)
            a+=1
            if a%3==0:
                raise ValueError("Couldn't initialise good parameters")
            if not finite:
                print "Fail: ", self.params_i,\
                        prior_bounds(self.params_i, self.params_l, self.params_u),\
                        prior_multim(self.params_i),\
                        prior_erfprecision(self.params_i, self.rngx_s, self.rngy_s), \
                        len(self.x_s)

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
            if self.runningL: print("\nInitparam likelihood: %.2f" % float(self.lnprob(params)))
            paramsOP = self.optimize(params, method, bounds)
            # Check likelihood for parameters
            lnlikeOP = self.lnlike(paramsOP)
            if self.runningL: print("\n %s: lnprob=%.0f, time=%d" % (method, self.lnprob(paramsOP), time.time()-start))
            params=paramsOP

        self.params_f_scaled = params.copy()
        if self.runscaling: params = self.unscaleParams(params)
        # Save evaluated parameters to internal values
        self.params_f = params.copy()

        return params

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

        # Set count to 0
        self.iter_count = 0

        # To clean up any warnings from optimize
        invalid = np.seterr()['invalid']
        divide = np.seterr()['divide']
        over = np.seterr()['over']
        np.seterr(invalid='ignore', divide='ignore', over='ignore')

        if type(method) is str:
            if method=='Stoch': optimizer = scipyStoch
            elif method=='Powell': optimizer = scipyOpt
            elif method=='Anneal': optimizer = scipyAnneal
            else: raise ValueError('Name of method not recognised.')
        else:
            optimizer = method
        kwargs = {'method':method, 'bounds':bounds}
        # result is the set of theta parameters which optimize the likelihood given x, y, yerr
        params, self.output = optimizer(self.nll, params)
        params = params.reshape(self.param_shape)
        # Potential to use scikit optimize
        #bounds = list(zip(self.params_l.ravel(), self.params_u.ravel()))
        #result = gp_minimize(self.nll, bounds)

        # To clean up any warnings from optimize
        np.seterr(invalid=invalid, divide=divide, over=over)
        if self.runningL: print("")

        return params

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
        if self.priorDF:
            function = lambda a, b: self.photoDF(*(a,b)) * self.distribution(params, a, b)
            model = self.photoDF( self.x_s, self.y_s ) * error_convolution(params, self.x_s, self.y_s, self.sig_xy_s)
        else:
            function = lambda a, b: self.distribution(params, a, b)
            model = error_convolution(params, self.x_s, self.y_s, self.sig_xy_s)

        # Point component of poisson log likelihood: contPoints \sum(\lambda(x_i))
        #model = function(*(self.x_s, self.y_s))
        contPoints = np.sum( np.log(model) )

        # Integral of the smooth function over the entire region
        contInteg = integrationRoutine(function, params, self.nComponents, self.rngx_s, self.rngy_s,
                                        self.x_2d, self.y_2d, integration='analyticApprox')
        #print bivGauss_analytical_approx(params, self.rngx_s, self.rngy_s), self.rngx_s, self.rngy_s

        lnL = contPoints - contInteg
        if self.runningL:
            sys.stdout.write("\ritern: %d, logL: %.2f, sum log(f(xi)): %.2f, integral: %.2f                " \
                            % (self.iter_count, lnL, contPoints, contInteg))
            sys.stdout.flush()
            self.iter_count += 1

        return lnL

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
        prior = prior_bounds(params, self.params_l, self.params_u)
        if not prior: return -np.inf

        # Test parameters against variance
        #if prior: prior = prior & sigma_bound(params, self.s_min)
        #if not prior: return -np.inf

        # Test parameters against boundary values
        prior = prior_multim(params)
        if not prior: return -np.inf

        # Test parameters against boundary values
        prior = prior_erfprecision(params, self.rngx_s, self.rngy_s)
        if not prior: return -np.inf

        # Prior on spectro distribution that it must be less than the photometric distribution
        if self.priorDF:
            function = lambda a, b: self.distribution(params, a, b)
            prior = prior & SFprior(function, *(self.rngx_s, self.rngy_s))
            if not prior: return -np.inf

        # All prior tests satiscied
        return 0.0

    def scaleParams(self, params_in):

        params = params_in.copy()
        print(params, self.mux, self.muy)
        print(params[:,[0,1]] -  [self.mux, self.muy]) / np.array([self.sx, self.sy])
        params[:,[0,1]] = (params[:,[0,1]] -  [self.mux, self.muy]) / np.array([self.sx, self.sy])
        params[:,[2,3]] *= 1/np.array([self.sx**2, self.sy**2])
        print(params)

        return params

    def unscaleParams(self, params_in):

        params = params_in.copy()
        params[:,[0,1]] = (params[:,[0,1]] * np.array([self.sx, self.sy])) +  [self.mux, self.muy]
        params[:,[2,3]] *= np.array([self.sx**2, self.sy**2])

        return params

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


def initParams(nComponents, rngx, rngy, priorDF, nstars=1, runscaling=True):

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
    mux_i, muy_i = np.random.rand(2, nComponents)
    if not runscaling: # Random means in range of system
        mux_i = mux_i * (rngx[1]-rngx[0]) + rngx[0]
        muy_i = muy_i * (rngy[1]-rngy[0]) + rngy[0]

    # Generate initial covariance matrix
    N_gen = 10
    X = np.random.rand(nComponents, 2, N_gen) / np.sqrt(N_gen)
    if not runscaling: # Standard deviations scaled to system
        X[:,0,:] *= np.sqrt(rngx[1]-rngx[0])
        X[:,1,:] *= np.sqrt(rngy[1]-rngy[0])
    sigma = np.matmul(X, X.transpose(0, 2, 1), np.zeros((nComponents, 2, 2)))

    # Weights sum to 1
    if priorDF: w_i = np.zeros(nComponents) + .1 / nComponents
    else: w_i = np.zeros(nComponents) + nstars/ nComponents

    # Lower and upper bounds on parameters
    # Mean at edge of range
    if runscaling:
        mux_l, mux_u = -np.inf, np.inf
        muy_l, muy_u = -np.inf, np.inf
    else:
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

    params_i = np.vstack((mux_i, sigma[:,0,0], muy_i, sigma[:,1,1], w_i, sigma[:,0,1])).T
    params_l = np.repeat([[mux_l, sxx_l, muy_l, syy_l, w_l, sxy_l],], nComponents, axis=0)
    params_u = np.repeat([[mux_u, sxx_u, muy_u, syy_u, w_u, sxy_u],], nComponents, axis=0)

    return params_i, params_l, params_u

def initParams_revamp(nComponents, rngx, rngy, priorDF, nstars=1, runscaling=True, l_min=0.1):

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
    mux_i, muy_i = np.random.rand(2, nComponents)
    mux_i = mux_i * (rngx[1]-rngx[0]) + rngx[0]
    muy_i = muy_i * (rngy[1]-rngy[0]) + rngy[0]

    # Generate initial covariance matrix
    l1_i, l2_i = np.sort(np.random.rand(2, nComponents), axis=0)
    l1_i = l1_i * np.min((rngx[1]-rngx[0], rngy[1]-rngy[0])) * 10 + l_min
    l2_i = l2_i * np.min((rngx[1]-rngx[0], rngy[1]-rngy[0])) * 10 + l_min

    # Initialise thetas
    th_i = np.random.rand(nComponents) * np.pi

    # Weights sum to 1
    w_i = np.random.rand(nComponents)/nComponents + .1 / nComponents
    w_l = 0
    w_u = np.inf
    if not priorDF: w_i *= nstars

    # Lower and upper bounds on parameters
    # Mean at edge of range
    mux_l, mux_u = -np.inf, np.inf
    muy_l, muy_u = -np.inf, np.inf
    # Zero standard deviation to inf
    l1_l, l2_l = l_min, l_min
    l1_u, l2_u = np.inf, np.inf #l_rng[1], l_rng[1] #rngx[1]-rngx[0], rngy[1]-rngy[0]
    # Covariance must be in range -1., 1.
    th_l, th_u = 0, np.pi


    params_i = np.vstack((mux_i, muy_i, l1_i, l2_i, th_i, w_i)).T
    params_l = np.repeat([[mux_l, muy_l, l1_l, l2_l, th_l, w_l],], nComponents, axis=0)
    params_u = np.repeat([[mux_u, muy_u, l1_u, l2_u, th_u, w_u],], nComponents, axis=0)

    params_i = params_i[params_i[:,3].argsort()]

    return params_i, params_l, params_u

def sigma_bound(params, s_min):

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
    # Construct covariance matrix into nComponent 2x2 arrays
    sigma = np.zeros((params.shape[0],2,2))
    sigma[:,[0,0,1,1],[0,1,0,1]] = params[:, [1,5,5,3]]
    try: eigvals = np.linalg.eigvals(sigma)
    except np.linalg.LinAlgError:
        print(params)
        print(sigma)
        raise ValueError('bad sigma...params:', params, 'sigma:', sigma)

    if np.sum(eigvals<=s_min) > 0: return False
    else: return True

def prior_bounds(params, params_l, params_u):

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
    total = np.sum(params <= params_l) + np.sum(params >= params_u)
    # prior True if all parameters within priors
    prior = total == 0

    return prior

def prior_multim(params):
    """
    prior_multim - Prior on eigenvalues and means of Gaussian mixture models
                    to remove some degenerate solutions (e.g. reordering components)
    """

    # Prior on the order of lambda values in one mixture component
    # Removes degeneracy between eigenvalue and angle
    l_order = np.sum(params[:,2]>params[:,3])
    # Prior on order of components.
    comp_order = np.sum( np.argsort(params[:,3]) != np.arange(params.shape[0]) )
    prior = l_order+comp_order == 0

    return prior

def prior_erfprecision(params, rngx, rngy):

    # shape 2,4 - xy, corners
    corners = np.array(np.meshgrid(rngx, rngy)).reshape(2, 4)
    # shape n,2,1 - components, xy, corners
    angle1 = np.array([np.sin(params[:,4]), np.cos(params[:,4])]).T[:,:,np.newaxis]
    angle2 = np.array([np.sin(params[:,4]+np.pi/2), np.cos(params[:,4]+np.pi/2)]).T[:,:,np.newaxis]
    # shape n,2,1 - components, xy, minmax
    mean = params[:,:2][:,:,np.newaxis]
    # shape n,4 - components, corners
    dl1 = np.sum( (corners - mean)*angle1 , axis=1)
    dl2 = np.sum( (corners - mean)*angle2 , axis=1)
    # shape 2,n,4 - axes, components, corners
    dl = np.stack((dl1, dl2))
    dl.sort(axis=2)
    # shape 2,n,2 - axes, components, extreme corners
    dl = dl[..., 0]

    # shape 2,n,2 - axes, components, extreme corners
    component_stds = params[:,2:4].T

    separation = np.abs(dl) / (np.sqrt(2) * component_stds)

    if np.sum(separation>20) > 0: return False
    else: return True


def SFprior(function, rngx, rngy, N=150):

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

    x_coords = np.linspace(rngx[0], rngx[1], N)
    y_coords = np.linspace(rngy[0], rngy[1], N)
    xx, yy = np.meshgrid(x_coords, y_coords)
    f_max = np.max( function(*(xx, yy)) )
    prior = not f_max>1

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

def bivGaussMix_vect(params, x, y):

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

    mu = np.array((params[:,0], params[:,2]))
    sigma = np.array([[params[:,1], params[:,5]], [params[:,5], params[:,3]]])
    weight= params[:,4]

    # Inverse covariance
    inv_cov = np.linalg.inv(sigma.transpose(2,0,1)).transpose(1,2,0)
    # Separation of X from mean
    X = np.moveaxis(np.repeat([X,], mu.shape[-1], axis=0), 0, -1) - mu
    # X^T * Sigma
    X_cov = np.einsum('mjn, jkn -> mkn', X, inv_cov)
    # X * Sigma * X
    X_cov_X = np.einsum('mkn, mkn -> mn', X_cov, X)
    # Exponential
    e = np.exp(-X_cov_X/2)

    # Normalisation term
    det_cov = np.linalg.det(sigma.transpose(2,0,1))
    norm = 1/np.sqrt( ((2*np.pi)**2) * det_cov)

    p = np.sum(weight*norm*e, axis=-1)
    return p.reshape(shape)

def bivGaussMix_vect_revamp(params, x, y):

    '''
    bivariateGauss - Calculation of bivariate Gaussian distribution.

    Parameters
    ----------
        params - arr of float - length 6 - [mux, muy, l1, l2, theta, weight]
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

    mu = params[:,:2]
    sigma = np.array([np.diag(a) for a in params[:,2:4]])
    R = rotation(params[:,4])
    weight= params[:,5]
    sigma = np.matmul(R, np.matmul(sigma, R.transpose(0,2,1)))

    # Inverse covariance
    inv_cov = np.linalg.inv(sigma)
    # Separation of X from mean
    X = np.moveaxis(np.repeat([X,], mu.shape[-2], axis=0), 0, -2) - mu

    # X^T * Sigma
    X_cov = np.einsum('mnj, njk -> mnk', X, inv_cov)
    # X * Sigma * X
    X_cov_X = np.einsum('mnk, mnk -> mn', X_cov, X)
    # Exponential
    e = np.exp(-X_cov_X/2)

    # Normalisation term
    det_cov = np.linalg.det(sigma)
    norm = 1/np.sqrt( ((2*np.pi)**2) * det_cov)

    p = np.sum(weight*norm*e, axis=-1)

    if np.sum(np.isnan(p))>0: print(params)

    return p.reshape(shape)

def error_convolution(params, x, y, sig_xy):

    shape = x.shape
    X = np.vstack((x.ravel(), y.ravel())).T

    # 1) Rotate S_component to x-y plane
    mu = params[:,:2]
    sigma = np.array([np.diag(a) for a in params[:,2:4]])
    R = rotation(params[:,4])
    weight= params[:,5]
    sigma = np.matmul(R, np.matmul(sigma, R.transpose(0,2,1)))

    # 2) Get Si + Sj for all stars_i, comonents_j
    sigma = sigma[np.newaxis, ...]
    sig_xy = sig_xy[:, np.newaxis, ...]
    sig_product = sigma + sig_xy

    # 3) Get mui - muj for all stars_i, comonents_j
    mu = mu[np.newaxis, ...]
    X = X[:, np.newaxis, ...]
    mu_product = mu - X

    # 4) Calculate Cij
    sig_product_inv, sig_product_det = inverse2x2(sig_product)
    # np.einsum('ijlm, ijm -> ijl', sig_product_inv, mu_product)
    exponent = -np.sum(mu_product * np.sum(sig_product_inv.transpose(2,0,1,3)*mu_product, axis=3).transpose(1,2,0), axis=2) / 2
    norm = 1/( 2*np.pi*np.sqrt(sig_product_det) )
    cij = norm*np.exp(exponent)

    # 6) Dot product with weights
    ci = np.sum(cij*params[:,5], axis=1)

    return ci
def inverse2x2(matrix):
    # Instead of np.linalg - This is so much faster!!!
    det = matrix[...,0,0]*matrix[...,1,1] - matrix[...,0,1]*matrix[...,1,0]
    #inv = matrix.copy()
    #inv[...,0,0] = matrix[...,1,1]
    #inv[...,1,1] = matrix[...,0,0]
    #inv[...,[0,1],[1,0]] *= -1
    #inv *= 1/np.repeat(np.repeat(det[...,np.newaxis,np.newaxis], 2, axis=-1), 2, axis=-2)

    inv = np.array([[matrix[...,1,1]/det, -matrix[...,0,1]/det],
                    [-matrix[...,1,0]/det, matrix[...,0,0]/det]]).transpose(2,3,0,1)

    return inv, det

def bivGaussMix_iter(params, x, y):

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

    p = np.sum(np.array([bivariateGauss(params[i,:], x, y) for i in range(params.shape[0])]), axis=0)

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

def covariance_scaling(sigxy, sx, sy):

    scaling = np.outer(np.array([sx, sy]), np.array([sx, sy]))
    return sigxy/scaling

def rotation(th):

    R = np.array([[np.cos(th), np.sin(th)],
                  [-np.sin(th), np.cos(th)]])

    return R.transpose(2,0,1)

"""
Optimizers
"""
def scipyOpt(function, params):

    bounds = None

    # result is the set of theta parameters which optimize the likelihood given x, y, yerr
    result = op.minimize(function, params.ravel(), method='Powell', bounds=bounds)
    params = result["x"]

    return params, result

def scipyAnneal(function, params):

    # result is the set of theta parameters which optimize the likelihood given x, y, yerr
    result = op.anneal(function, params.ravel())
    params = result["x"]

    return params, result

def scipyStoch(function, params):

    # result is the set of theta parameters which optimize the likelihood given x, y, yerr
    result = op.basinhopping(function, params.ravel(), niter=1)
    params = result["x"]

    return params, result

def emcee_opt(function, params, niter=2000, file_loc=''):

    pshape =params.shape
    foo = lambda pars: -function(pars.reshape(pshape))

    nwalkers=int(params.shape[0]*2.5)
    ndim=len(params.flatten())

    p0 = np.array([initParams_revamp(params.shape[0], [-20,20],[-20,20],
                    False,nstars=2000,runscaling=True,l_rng=[0.1, 1.])[0].flatten() for i in range(nwalkers)])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, foo)
    # Run emcee
    _=sampler.run_mcmc(p0, niter)

    sampler.chain

    # Retrieve results
    nburn = niter/2
    burnt_values = sampler.chain[:,nburn:,:]
    burnt_values = burnt_values.reshape(-1, burnt_values.shape[-1])

    median = np.median(burnt_values, axis=0)

    lp = sampler.lnprobability
    index = np.unravel_index(np.argmax(lp), lp.shape)
    median = sampler.chain[index[0], index[1], :]

    if savefigs != '':
        import corner

        plt.figure( figsize=(10*params.shape[0], 60) )
        axes = plt.subplots(params.shape[1], params.shape[0])
        for i in xrange(median.shape[0]):
            for j in xrange(median.shape[1]):
                plt.sca(axes[i,j])
                for k in range(nwalkers):
                    plt.plot(np.arange(sampler.chain.shape[1]), sampler.chain[k,:,i], color="0.2", linewidth=0.1)
                burnt = sampler.chain[...,i].flatten()
                mean = np.mean(burnt)
                median = np.median(burnt)
                plt.plot([0,sampler.chain.shape[1]], [mean, mean], label='mean after burn in')
                plt.plot([0,sampler.chain.shape[1]], [median, median], label='median after burn in')
                plt.legend()
                plt.title("Dimension {0:d}".format(i))
                plt.savefig(file_loc, bbox_inches='tight')

        plt.figure( figsize=(20, 20) )
        fig = corner.corner(burnt_values, quantiles=[0.5], show_titles=True)
        plt.savefig(file_loc, bbox_inches='tight')


    return median, sampler


"""
INTEGRATION ROUTINES
"""
def integrationRoutine(function, param_set, nComponents, rngx, rngy, x_2d, y_2d, integration = "trapezium"):

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
    elif integration == "trapezium": contInteg = numericalIntegrate_precompute(function, x_2d, y_2d)
    elif integration == "analyticApprox": contInteg = bivGauss_analytical_approx(param_set, rngx, rngy)
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

def numericalIntegrate_mesh(function, rngx, rngy, Nx_int=250, Ny_int=250):

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

    x_2d, y_2d = np.meshgrid(x_coords, y_coords)
    z_2d = function(*(x_2d, y_2d))

    volume1 = ( (z_2d[:-1, :-1] + z_2d[1:, 1:])/2 ) * dx * dy
    volume2 = ( (z_2d[:-1, 1:] + z_2d[1:, :-1])/2 ) * dx * dy
    integral = ( np.sum(volume1.flatten()) + np.sum(volume2.flatten()) ) /2

    return integral

def numericalIntegrate_precompute(function, x_2d, y_2d):

    z_2d = function(*(x_2d, y_2d))

    dx = x_2d[1:, 1:] - x_2d[1:, :-1]
    dy = y_2d[1:, 1:] - y_2d[:-1, 1:]

    volume1 = ( (z_2d[:-1, :-1] + z_2d[1:, 1:])/2 ) * dx * dy
    volume2 = ( (z_2d[:-1, 1:] + z_2d[1:, :-1])/2 ) * dx * dy
    integral = ( np.sum(volume1.flatten()) + np.sum(volume2.flatten()) ) /2

    return integral

def bivGauss_analytical_approx2(params, rngx, rngy):

    dl1 = find_dls(rngx, rngy, params)
    dl2 = find_dls(rngx, rngy, params, rotate=np.pi/2)

    # Assuming indices are xmin, ymin, xmax, ymax
    dl_asc1 = np.sort(dl1,axis=1)
    dl_asc2 = np.sort(dl2,axis=1)
    dlf1 = np.zeros((params.shape[0], 2))
    dlf2 = np.zeros((params.shape[0], 2))

    boundary_index_1a = np.argmin(np.abs(dl1), axis=1)
    boundary_index_1b = boundary_index_1a - 2 # To get opposite boundary
    boundary_index_2a = np.argmin(np.abs(dl2), axis=1)
    boundary_index_2b = boundary_index_2a - 2 # To get opposite boundary

    ncross1 = np.sum(dl1>0, axis=1)
    ncross2 = np.sum(dl2>0, axis=1)
    # Condition1 - Ellipse lies within boundaries
    con = (ncross1==2)&(ncross2==2)
    dlf1[con] = np.array([np.abs(dl1[con][np.arange(dlf1[con].shape[0]),boundary_index_1a[con]-1]),
                          np.abs(dl1[con][np.arange(dlf1[con].shape[0]),boundary_index_1b[con]-1])]).T
    dlf2[con] = np.array([np.abs(dl2[con][np.arange(dlf2[con].shape[0]),boundary_index_2a[con]-1]),
                          np.abs(dl2[con][np.arange(dlf2[con].shape[0]),boundary_index_2b[con]-1])]).T
    con = (ncross1==3)
    dlf1[con] = np.array([np.zeros(np.sum(con)), np.abs(dl_asc1[con][:,3])]).T
    con = (ncross2==3)
    dlf2[con] = np.array([np.zeros(np.sum(con)), np.abs(dl_asc2[con][:,3])]).T
    con = (ncross1==1)
    dlf1[con] = np.array([np.zeros(np.sum(con)), np.abs(dl_asc1[con][:,0])]).T
    con = (ncross2==1)
    dlf2[con] = np.array([np.zeros(np.sum(con)), np.abs(dl_asc2[con][:,0])]).T
    con = (ncross1==2)&((ncross2==0)|(ncross2==4))
    dlf1[con] = np.array([np.abs(dl_asc1[con][:,0]), np.abs(dl_asc1[con][:,3])]).T
    dlf2[con] = np.array([np.abs(dl2[con][np.arange(dlf2[con].shape[0]),boundary_index_2a[con]]),
                          np.abs(dl2[con][np.arange(dlf2[con].shape[0]),boundary_index_2b[con]])]).T
    con = ((ncross1==0)|(ncross1==4))&(ncross2==2)
    dlf1[con] = np.array([np.abs(dl1[con][np.arange(dlf1[con].shape[0]),boundary_index_1a[con]]),
                          np.abs(dl1[con][np.arange(dlf1[con].shape[0]),boundary_index_1b[con]])]).T
    dlf2[con] = np.array([np.abs(dl_asc2[con][:,0]), np.abs(dl_asc2[con][:,3])]).T

    dl = np.stack((dlf1, dlf2))

    erfs = spec.erf( dl / (np.sqrt(2) * np.repeat([params[:,2:4],], 2, axis=0)) ).transpose(1,2,0) / 2

    #comp_integral = np.zeros(erfs.shape[:2])
    #comp_integral[erfs[:,0,:]<0] = 0.5-erfs[:,1,:][erfs[:,0,:]<0]
    #comp_integral[erfs[:,0,:]>0] = np.sum(erfs, axis=1)[erfs[:,0,:]>0]
    comp_integral = np.sum(erfs, axis=1)
    comp_integral = np.prod(comp_integral, axis=1)
    integral = np.sum(comp_integral*params[:,5])

    return integral

def find_dls(rngx, rngy, params, rotate=0.):

    # shape 2,2 - xy, minmax
    rngxy = np.array([rngx, rngy])
    # shape n,2,1 - components, xy, minmax
    angle = np.array([np.sin(params[:,4]+rotate), np.cos(params[:,4]+rotate)]).T[:,:,np.newaxis]
    # shape n,2,1 - components, xy, minmax
    mean = params[:,:2][:,:,np.newaxis]

    # shape n,2,2 - components, xy, minmax
    dl = (rngxy - mean)/angle
    # at this point I know which boundaries belong to which coordinates
    # Can I chose which boundary indices to use here then index them lower down keeping the con options???
    # Get argmin of abs values, take index of argmin+2 as second boundary
    # shape n, 4 - components, xyminmax
    dl = dl.transpose(0,2,1).reshape(dl.shape[0], -1)

    """
    #NOT sure if this works
    boundary_index_1 = np.argmin(np.abs(dl), axis=1)
    boundary_index_2 = boundary_index_1 - 2 # To get opposite boundary
    # Get rid of dl.sort
    dltest = dl.copy()
    # Assuming indices are xmin, ymin, xmax, ymax

    dl.sort(axis=1)

    dlf = np.zeros((params.shape[0], 2))
    con = np.sum(dl>0, axis=1)==4
    dlf[con] = np.array([np.zeros(np.sum(con))-1, dl[con][:,0]]).T
    con = np.sum(dl>0, axis=1)==3
    dlf[con] = np.array([np.zeros(np.sum(con))-1, dl[con][:,1]]).T
    con = np.sum(dl>0, axis=1)==2
    dlf[con] = np.array([-dl[con][:,1],  dl[con][:,2]]).T
    dlf[con] = np.array([np.abs(dltest[con][np.arange(dlf[con].shape[0]),boundary_index_1[con]]),
                        np.abs(dltest[con][np.arange(dlf[con].shape[0]),boundary_index_2[con]])]).T
    con = np.sum(dl>0, axis=1)==1
    dlf[con] = np.array([np.zeros(np.sum(con))-1, -dl[con][:,2]]).T
    con = np.sum(dl>0, axis=1)==0
    dlf[con] = np.array([np.zeros(np.sum(con))-1, -dl[con][:,3]]).T"""

    return dl

def bivGauss_analytical_approx(params, rngx, rngy):

    # shape 2,4 - xy, corners
    corners = np.array(np.meshgrid(rngx, rngy)).reshape(2, 4)
    # shape n,2,1 - components, xy, corners
    angle1 = np.array([np.cos(params[:,4]), -np.sin(params[:,4])]).T[:,:,np.newaxis]
    angle2 = np.array([np.sin(params[:,4]), np.cos(params[:,4])]).T[:,:,np.newaxis]
    # shape n,2,1 - components, xy, minmax
    mean = params[:,:2][:,:,np.newaxis]

    #print 'Corners: ', corners
    # shape n,4 - components, corners
    dl1 = np.sum( (corners - mean)*angle1 , axis=1)
    dl2 = np.sum( (corners - mean)*angle2 , axis=1)
    # shape 2,n,4 - axes, components, corners
    dl = np.stack((dl1, dl2))
    #print 'Dl: ', dl[:,1,:]
    dl.sort(axis=2)
    # shape 2,n,2 - axes, components, extreme corners
    dl = dl[..., [0,-1]]
    #print 'Dl minmax: ', dl[:,1,:]

    # shape 2,n,2 - axes, components, extreme corners
    component_vars = np.repeat([params[:,2:4],], 2, axis=0).transpose(2,1,0)
    #print 'stds: ', component_vars[:,1,:]
    # Use erfc on absolute values to avoid high value precision errors
    sign = dl/np.abs(dl)
    erfs = spec.erfc( np.abs(dl) / (np.sqrt(2)*np.sqrt(component_vars) ) )
    #print 'erfs: ', erfs[:,1,:]
    erfs = erfs*sign
    #print 'ratio: ', dl / (np.sqrt(2) * component_stds)
    #print 'erfs: ', erfs
    #print 'sigmas: ', np.repeat([params[:,2:4],], 2, axis=0).transpose(2,1,0)
    #print 'erfs: ', erfs[:,1,:]

    # Sum integral lower and upper bounds
    comp_integral = (np.abs(erfs[...,1]-erfs[...,0]) + np.abs(sign[...,0] - sign[...,1])) / 2
    #print 'comp: ', comp_integral[:,1]
    # Product the axes of the integrals
    comp_integral = np.prod(comp_integral, axis=0)
    #return comp_integral
    # Sum weighted Gaussian components
    integral = np.sum(comp_integral*params[:,5])

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

"""
TESTS
"""
def singleGaussianSample(mu, sigma, N = 1000):

    # Generate 2D sample with mean=0, std=1
    sample = np.random.normal(size=(2, N))

    # Convert sample to given mean and covariance
    A = np.linalg.cholesky(sigma)
    sample = mu + np.matmul(A, sample).T

    return sample




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

        return resul
