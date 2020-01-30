
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
import emcee
#from skopt import gp_minimize
import sys, os, time
from mpmath import *

from sklearn import mixture
from sklearn.cluster import KMeans
from tqdm import tqdm_notebook as tqdm

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

    def __init__(self, x=np.zeros(0), y=np.zeros(0), sig_xy=None,
                nComponents=0, rngx=(0,1), rngy=(0,1), runscaling=True, runningL=True, s_min=0.1,
                photoDF=None, priorDF=False):

        # Iteration number to update
        self.iter_count = 0

        # Name of the model to used for reloading from dictionary
        self.modelname = self.__class__.__name__

        # Distribution from photometric survey for calculation of SF
        self.photoDF = photoDF
        self.priorDF = priorDF

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

        # Method used for integration
        self.integration='trapezium'

        # Coordinate covariance matrix
        if sig_xy is None:
            z_ = np.zeros(len(x))
            sig_xy = np.array([[z_, z_],[z_, z_]]).transpose(2,0,1)
        self.sig_xy = sig_xy

        self.runscaling = runscaling
        # Not run when loading class from dictionary
        if runscaling:
            # Real space parameters
            self.x = x.copy()
            self.y = y.copy()
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
        self.distribution = bivGaussMixture

        # Print out likelihood values as calculated
        self.runningL = runningL

        Nx_int, Ny_int = (250,250)
        x_coords = np.linspace(self.rngx_s[0], self.rngx_s[1], Nx_int)
        y_coords = np.linspace(self.rngy_s[0], self.rngy_s[1], Ny_int)
        self.x_2d, self.y_2d = np.meshgrid(x_coords, y_coords)

        if self.priorDF:
            # Calculate Gaussian distributions from product of scaled DF and scaled star positions
            self.params_df = self.scaleParams(self.photoDF.params_f, dfparams=True)
            function = lambda a, b: self.distribution(self.params_df, a, b)
            #if self.runningL:
            #    print 'DF integral = ', numericalIntegrate_precompute(function, self.x_2d, self.y_2d)
            self.ndf = len(self.photoDF.x)


        else: self.ndf = None

    def __call__(self, x, y, components=None, params=None):

        '''
        __call__ - Returns the value of the smooth GMM distribution at the points x, y

        Parameters
        ----------
            x, y - float or np.array of floats
                - x and y coordinates of points at which to take the value of the GMM
                - From input - x is magnitude, y is colour

            components=None:
                - List of components to check for distribution values

            params=None:
                - The parameters on which the model will be evaluatedself.
                - If None, params_f class attribute will be used

        Returns
        -------
            GMMval: float or np.array of floats
                - The value of the GMM at coordinates x, y
        '''
        #
        if params is None: params=self.params_f.copy()

        # Scale x and y to correct region - Currently done to params_f - line 371  - but could change here instead
        #x, y = feature_scaling(x, y, self.mux, self.muy, self.sx, self.sy)
        #rngx, rngy = feature_scaling(np.array(self.rngx), np.array(self.rngy), self.mux, self.muy, self.sx, self.sy)
        rngx, rngy = np.array(self.rngx), np.array(self.rngy)

        # Value of coordinates x, y in the Gaussian mixture model
        if components is None: components = np.arange(self.nComponents)
        GMMval = self.distribution(params[components, :], x, y)

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

    def optimizeParams(self, method="Powell", init="random"):

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
        if self.runningL: print('init, ', init)

        while not finite:
            if not init=="reset":
                self.params_i, self.params_l, self.params_u = \
                    initParams(self.nComponents, self.rngx_s, self.rngy_s, self.priorDF,
                                    nstars=len(self.x_s), ndf=self.ndf, runscaling=self.runscaling, l_min=self.s_min)
            if init=="kmeans":
                # K-Means clustering of initialisation
                if not self.priorDF: params_km = kmeans(np.vstack((self.x_s, self.y_s)).T, self.nComponents)
                elif self.priorDF:
                    weights = 1/self.distribution(self.params_df, self.x_s, self.y_s)
                    params_km = kmeans(np.vstack((self.x_s, self.y_s)).T, self.nComponents, weights=weights, ndf=self.ndf)
                self.params_i[:,[0,1,5]] = params_km[:,[0,1,5]]
                self.params_i[self.params_i<self.params_l] = self.params_l[self.params_i<self.params_l]*1.01
                self.params_i = self.params_i[self.params_i[:,0].argsort()]

            if init=='reset':
                # Using current final parameters to initialise parameters
                params = self.params_f_scaled
            else:
                params = self.params_i.copy()

            if self.runningL: print('initial parameters', params)
            self.param_shape = params.shape
            lnp = self.lnprob(params)
            finite = np.isfinite(lnp)
            a+=1
            if a%3==0:
                raise ValueError("Couldn't initialise good parameters")
            if not finite:
                print("Fail: ", self.params_i,\
                        prior_bounds(self.params_i, self.params_l, self.params_u),\
                        prior_multim(self.params_i),\
                        prior_erfprecision(self.params_i, self.rngx_s, self.rngy_s), \
                        len(self.x_s))

        start = time.time()
        bounds = None
        # Run scipy optimizer
        if self.runningL: print("\nInitparam likelihood: %.2f" % float(self.lnprob(params)))
        paramsOP = self.optimize(params, method, bounds)
        # Check likelihood for parameters
        lnlikeOP = self.lnlike(paramsOP)
        if self.runningL: print("\n %s: lnprob=%.0f, time=%d" % (method, self.lnprob(paramsOP), time.time()-start))
        params=paramsOP

        if not method=='emceeBall':
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
            elif method=='emceeBall': optimizer = lambda a, b: emcee_ball(a, b, params_l=self.params_l, params_u=self.params_u)
            else: raise ValueError('Name of method not recognised.')
        else:
            optimizer = method
        kwargs = {'method':method, 'bounds':bounds}
        # result is the set of theta parameters which optimize the likelihood given x, y, yerr
        test = self.nll(params)
        if self.runningL: print(r'\nInit  lnl: ', test, r'\n')
        params, self.output = optimizer(self.nll, params)#, pl = self.params_l[0,:], pu = self.params_u[0,:])
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
            # Create copies to prevent overwrite problems
            params1, params2 = params.copy(), self.params_df.copy()
            function = lambda a, b: self.distribution(params1, a, b) \
                                    * self.distribution(params2, a, b)
            #model = self.norm_df_spec*error_convolution(params, self.mu_df_spec[:,0], self.mu_df_spec[:,1], self.sig_df_spec)
            params = gmm_product_p(params, self.params_df)
            params = params.reshape(-1, params.shape[-1])
            model = error_convolution(params, self.x_s, self.y_s, self.sig_xy_s)
        else:
            function = lambda a, b: self.distribution(params, a, b)
            model = error_convolution(params, self.x_s, self.y_s, self.sig_xy_s)
            #model = self.distribution(params, self.x_s, self.y_s)

        # Point component of poisson log likelihood: contPoints \sum(\lambda(x_i))
        #model = function(*(self.x_s, self.y_s))
        contPoints = np.sum( np.log(model) )

        # Integral of the smooth function over the entire region
        contInteg = integrationRoutine(function, params, self.nComponents, self.rngx_s, self.rngy_s,
                                        self.x_2d, self.y_2d, integration=self.integration)
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
        if self.integration=='analyticApprox':
            prior = prior_erfprecision(params, self.rngx_s, self.rngy_s)
            if not prior: return -np.inf

        # Prior on spectro distribution that it must be less than the photometric distribution
        if self.priorDF:
            function = lambda a, b: self.distribution(params, a, b)
            prior = prior & SFprior(function, self.x_2d, self.y_2d)
            if not prior: return -np.inf

        # All prior tests satiscied
        return 0.0

    def scaleParams(self, params_in, dfparams=False):

        params = params_in.copy()
        params[:,[0,1]] = (params[:,[0,1]] -  [self.mux, self.muy]) / np.array([self.sx, self.sy])

        sigma = np.array([np.diag(a) for a in params[:,2:4]])
        R = rotation(params[:,4])
        sigma = np.matmul(R, np.matmul(sigma, R.transpose(0,2,1)))

        sigma[:,[0,1],[0,1]] *= 1/np.array([self.sx**2, self.sy**2])
        sigma[:,[0,1],[1,0]] *= 1/(self.sx*self.sy)

        eigvals, eigvecs = np.linalg.eig(sigma)
        # Line eigvecs up with eigvals
        i = eigvals.argsort(axis=1)
        j = np.repeat([np.arange(eigvals.shape[0]),], eigvals.shape[1], axis=0).T
        eigvecs = eigvecs[j, i, :]
        eigvals = eigvals[j, i]

        params[:,[2,3]] = eigvals #np.sort(eigvals, axis=1)
        th = np.arctan2(eigvecs[:,0,1], eigvecs[:,0,0])
        params[:,4] = th

        if self.priorDF & (not dfparams):
            params[:,5] /= (self.sx*self.sy)

        return params

    def unscaleParams(self, params_in, dfparams=False):

        params = params_in.copy()
        params[:,[0,1]] = (params[:,[0,1]] * np.array([self.sx, self.sy])) +  [self.mux, self.muy]

        sigma = np.array([np.diag(a) for a in params[:,2:4]])
        R = rotation(params[:,4])
        sigma = np.matmul(R, np.matmul(sigma, R.transpose(0,2,1)))

        sigma[:,[0,1],[0,1]] *= np.array([self.sx**2, self.sy**2])
        sigma[:,[0,1],[1,0]] *= self.sx*self.sy

        eigvals, eigvecs = np.linalg.eig(sigma)

        i = eigvals.argsort(axis=1)
        j = np.repeat([np.arange(eigvals.shape[0]),], eigvals.shape[1], axis=0).T
        eigvecs = eigvecs[j, i, :]
        eigvals = eigvals[j, i]

        params[:,[2,3]] = eigvals #np.sort(eigvals, axis=1)
        th = np.arctan2(eigvecs[:,0,1], eigvecs[:,0,0])
        params[:,4] = th

        if self.priorDF & (not dfparams):
            params[:,5] *= (self.sx*self.sy)

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
        This needs updating!!

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


def initParams(nComponents, rngx, rngy, priorDF, nstars=1, ndf=None, runscaling=True, l_min=0.1):

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

    # l_min to allow erf calculation:
    half_diag = np.sqrt((rngx[1]-rngx[0])**2 + (rngy[1]-rngy[0])**2)/2
    #l_min = min(half_diag*np.sqrt(2)/24, np.min((rngx[1]-rngx[0], rngy[1]-rngy[0]))/10)
    #l_min = 0.01
    if priorDF: l_min = np.sqrt((rngx[1]-rngx[0])*(rngy[1]-rngy[0])/nstars)
    if not priorDF: l_min = np.sqrt((rngx[1]-rngx[0])*(rngy[1]-rngy[0])/nstars)/10
    #l_min = half_diag/4
    # Generate initial covariance matrix
    l1_i, l2_i = np.sort(np.random.rand(2, nComponents), axis=0)
    l1_i = l1_i * half_diag + l_min
    l2_i = l2_i * half_diag + l_min

    # Initialise thetas
    th_i = np.random.rand(nComponents) * np.pi

    # Weights sum to 1
    w_l = 1./(5*nComponents) # 5 arbitrarily chosen
    w_i = np.random.rand(nComponents)*(2./nComponents - w_l) + w_l
    w_u = 10. # Arbitrarily chosen
    if priorDF:
        w_l *= float(nstars)/ndf
        w_i *= float(nstars)/ndf
    if not priorDF:
        w_l *= nstars
        w_i *= nstars
        w_u *= nstars

    # Lower and upper bounds on parameters
    # Mean at edge of range
    mux_l, mux_u = rngx[0], rngx[1]
    muy_l, muy_u = rngy[0], rngy[1]
    # Zero standard deviation to inf
    l1_l, l2_l = l_min, l_min
    l1_u, l2_u = np.min((rngx[1]-rngx[0], rngy[1]-rngy[0]))*30, np.max((rngx[1]-rngx[0], rngy[1]-rngy[0]))*30
    #, np.inf, np.inf #l_rng[1], l_rng[1] #

    th_l, th_u = 0, np.pi


    params_i = np.vstack((mux_i, muy_i, l1_i, l2_i, th_i, w_i)).T
    params_l = np.repeat([[mux_l, muy_l, l1_l, l2_l, th_l, w_l],], nComponents, axis=0)
    params_u = np.repeat([[mux_u, muy_u, l1_u, l2_u, th_u, w_u],], nComponents, axis=0)

    params_i = params_i[params_i[:,0].argsort()]

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
    comp_order = np.sum( np.argsort(params[:,0]) != np.arange(params.shape[0]) )
    #comp_order=0
    prior = l_order+comp_order == 0
    #print 'order: ', l_order, comp_order

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

    if np.sum(separation>25) > 0: return False
    else: return True

def SFprior(function, xx, yy):

    '''
    SFprior - The selection function has to be between 0 and 1 everywhere.
        - informative prior

    Parameters
    ----------
        function - function
            - The selection function interpolant over the region, R.

        xx, yy:

    Returns
    -------
        prior - bool
            - True if all points on GMM are less than 1.
            - Otherwise False
    '''

    f_max = np.max( function(*(xx, yy)) )
    prior = not f_max>1

    return prior


def bivGaussMix_vect(params, x, y):

    '''
    bivGaussMix_vect - Calculation of bivariate Gaussian distribution.

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
    X_ext = X[...,np.newaxis]
    inv_cov = inv_cov[np.newaxis,...]
    X_cov = X_ext*inv_cov
    X_cov = X_cov[...,0,:]+X_cov[...,1,:]
    # X * Sigma * X
    X_cov_X = X_cov*X
    X_cov_X = X_cov_X[:,:,0]+X_cov_X[:,:,1]
    # Exponential
    e = np.exp(-X_cov_X/2)

    # Normalisation term
    det_cov = np.linalg.det(sigma)
    norm = 1/np.sqrt( ((2*np.pi)**2) * det_cov)

    p = np.sum(weight*norm*e, axis=-1)

    #if np.sum(np.isnan(p))>0: print(params)

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
def gmm_product(mu1, mu2, sig1, sig2):

    sig1_i = inverse2x2(sig1)[0]
    sig2_i = inverse2x2(sig2)[0]
    sig3 = inverse2x2(sig1_i + sig2_i)[0]

    mu1 = np.repeat(mu1, mu2.shape[0], axis=0)[...,np.newaxis]
    mu2 = np.repeat(mu2, mu1.shape[1], axis=1)[...,np.newaxis]
    mu1 = np.repeat(mu1, 2, axis=3)
    mu2 = np.repeat(mu2, 2, axis=3)

    #mu3 = np.einsum('nmij, nmj -> nmi', np.matmul(sig3, sig1_i), mu1) + \
        #        np.einsum('nmij, nmj -> nmi', np.matmul(sig3, sig2_i), mu2)

    mu3 = np.matmul(np.matmul(sig3, sig1_i), mu1, out=np.zeros(sig3.shape)) + \
        np.matmul(np.matmul(sig3, sig2_i), mu2, out=np.zeros(sig3.shape))
    mu3 = mu3[...,0]

    return mu3, sig3
def gmm_product_p(params1, params2):

    sig1 = np.array([np.diag(a) for a in params1[:,2:4]])
    R = rotation(params1[:,4])
    sig1 = np.matmul(R, np.matmul(sig1, R.transpose(0,2,1)))

    sig2 = np.array([np.diag(a) for a in params2[:,2:4]])
    R = rotation(params2[:,4])
    sig2 = np.matmul(R, np.matmul(sig2, R.transpose(0,2,1)))

    mu1 = params1[:,:2]
    mu2 = params2[:,:2]

    sig1 = sig1[np.newaxis, ...]
    sig2 = sig2[:, np.newaxis, ...]
    mu1 = mu1[np.newaxis, ...]
    mu2 = mu2[:, np.newaxis, ...]
    mu3, sig3 = gmm_product(mu1, mu2, sig1, sig2)

    sig_norm_i, sig_norm_det = inverse2x2(sig1+sig2)
    mu_norm = mu1-mu2
    exponent = -np.sum(mu_norm * np.sum(sig_norm_i.transpose(2,0,1,3)*mu_norm, axis=3).transpose(1,2,0), axis=2) / 2
    norm = 1/( 2*np.pi*np.sqrt(sig_norm_det) )
    cij = norm*np.exp(exponent)

    w1 = params1[:,[5]][np.newaxis, ...]
    w2 = params2[:,[5]][:, np.newaxis, ...]
    cij = cij[..., np.newaxis]
    w3 = w1*w2*cij

    eigvals, eigvecs = np.linalg.eig(sig3)
    th3 = np.arctan2(eigvecs[...,0,1], eigvecs[...,0,0])[...,np.newaxis]

    params3 = np.concatenate((mu3, eigvals, th3, w3), axis=2)

    return params3

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

    p0 = np.array([initParams(params.shape[0], [-20,20],[-20,20],
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

def emcee_ball(function, params, params_l=None, params_u=None, niter=2000):
    print('emcee with %d iterations...' % niter)

    pshape =params.shape
    foo = lambda pars: -function(pars.reshape(pshape))

    ndim=len(params.flatten())
    nwalkers=ndim*2

    p0 = np.repeat([params,], nwalkers, axis=0)
    p0 = np.random.normal(loc=p0, scale=np.abs(p0/500))
    p0[0,:] = params

    # Reflect out of bounds parameters back into the prior boundaries
    # Lower bound
    pl = np.repeat([params_l,], nwalkers, axis=0)
    lb = p0 < pl
    p0[lb] = pl[lb] + pl[lb] - p0[lb]
    # Upper bound
    pu = np.repeat([params_u,], nwalkers, axis=0)
    ub = p0 > pu
    p0[ub] = pu[ub] + pu[ub] - p0[ub]
    # Order eigenvalues
    p0[:,:,2:4] = np.sort(p0[:,:,2:4], axis=2)
    p0[:,:,0] = np.sort(p0[:,:,0], axis=1)
    #sort_i = p0[:,:,0].argsort(axis=1)
    #sort_j = np.repeat([np.arange(p0.shape[0]),], p0.shape[1], axis=0).T
    #p0 = p0[sort_j, sort_i, :]

    p0 = p0.reshape(nwalkers, -1)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, foo)
    # Run emcee
    _=sampler.run_mcmc(p0, niter)

    # Retrieve results
    nburn = niter/2
    burnt_values = sampler.chain[:,nburn:,:]
    burnt_values = burnt_values.reshape(-1, burnt_values.shape[-1])

    median = np.median(burnt_values, axis=0)

    lp = sampler.lnprobability
    index = np.unravel_index(np.argmax(lp), lp.shape)
    median = sampler.chain[index[0], index[1], :]

    return median, sampler

def kmeans(sample, nComponents, n_iter=10, max_iter=100, weights=None, ndf=None):

    weighted_kde = not weights is None
    if weights is None: weights=np.ones(len(sample))

    params = np.zeros((nComponents, 6))
    from sklearn.cluster import KMeans

    kmc = KMeans(nComponents, n_init=n_iter, max_iter=max_iter)
    kmc.fit(sample, sample_weight=weights)

    means = kmc.cluster_centers_

    s0 = sample[kmc.labels_==0]
    for i in xrange(nComponents):
        sample_i = sample[kmc.labels_==i]
        delta = sample_i - means[i]
        sigma = np.matmul(delta.T, delta)/delta.shape[0]

        eigvals, eigvecs = np.linalg.eig(sigma)
        eigvecs = eigvecs[np.argsort(eigvals),:]
        eigvals = np.sort(eigvals)
        theta = np.arctan2(eigvecs[0,1], eigvecs[0,0])
        if theta<0: theta+=np.pi

        if not weighted_kde: w = sample_i.shape[0]
        else:
            w = np.sum(weights[kmc.labels_==i])/np.sum(weights)
            w *= float(sample.shape[0])/ndf

        params[i,:] = np.array([means[i,0], means[i,1], eigvals[0], eigvals[1], theta, w])

    return params

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
    else: raise ValueError('No integration routine "%s"' % integration)

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

    #print 'Delta', corners-mean

    #print 'Corners: ', corners
    # shape n,4 - components, corners
    dl1 = np.sum( (corners - mean)*angle1 , axis=1)
    dl2 = np.sum( (corners - mean)*angle2 , axis=1)
    # shape 2,n,4 - axes, components, corners
    dl = np.stack((dl1, dl2))
    #print 'Dl: ', dl[:,0,:]
    dl.sort(axis=2)
    # shape 2,n,2 - axes, components, extreme corners
    dl = dl[..., [0,-1]]
    #print 'Dl minmax: ', dl

    # shape 2,n,2 - axes, components, extreme corners
    component_vars = np.repeat([params[:,2:4],], 2, axis=0).transpose(2,1,0)
    #print 'vars: ', component_vars
    # Use erfc on absolute values to avoid high value precision errors
    erfs = spec.erfc( dl / (np.sqrt(2)*np.sqrt(component_vars) ) )
    #print 'erfs: ', erfs
    #print 'ratio: ', dl / (np.sqrt(2) * component_stds)
    #print 'erfs: ', erfs
    #print 'sigmas: ', np.repeat([params[:,2:4],], 2, axis=0).transpose(2,1,0)
    #print 'erfs: ', erfs[:,1,:]

    # Sum integral lower and upper bounds
    comp_integral = np.abs(erfs[...,1]-erfs[...,0]) / 2
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



"""
BayesianGaussianMixture and TNC methods
"""
class BGM_TNC():

    '''
    GaussianEM - Class for calculating bivariate Gaussian mixture model which best fits
                 the given poisson point process data.

    Parameters
    ----------
        x, y - np.array of floats
            - x and y coordinates for the points generated via poisson point process from
            the smooth model


        rngx, rngy - tuple of floats
            - Upper and lower bounds on x and y of the survey

    Functions
    ---------
        __call__ - Returns the value of the smooth GMM distribution at the points x, y
        optimizeParams - Vary the parameters of the distribution using the given method to optimize the
                        poisson likelihood of the distribution
        optimize
        lnprob - ln of the posterior probability of the distribution given the parameters.
               - posterior probability function is proportional to the prior times the likelihood
               - lnpost = lnprior + lnlike
        lnprior - The test of the parameters of the Gaussian Mixture Model against the specified prior values
    '''

    def __init__(self, x=np.zeros(0), y=np.zeros(0), sig_xy=None,
                rngx=(0,1), rngy=(0,1), runscaling=True, scales=None, runningL=False,
                photoDF=None, priorDF=False, prior_sfBounds=None):

        # Iteration number to update
        self.iter_count = 0

        # Name of the model to used for reloading from dictionary
        self.modelname = self.__class__.__name__

        # Distribution from photometric survey for calculation of SF
        self.photoDF = photoDF
        self.priorDF = priorDF

        # Starting values for parameters
        self.params_i = None
        # Final optimal values for parameters
        self.params_f = None
        # Shape of parameter set (number of components x parameters per component)
        self.param_shape = ()

        # Boundaries used to generate NIW priors
        self.prior_sfBounds = prior_sfBounds
        if runningL: print('Prior boundaries: ', prior_sfBounds)

        # Coordinate covariance matrix
        if sig_xy is None:
            z_ = np.zeros(len(x))
            sig_xy = np.array([[z_, z_],[z_, z_]]).transpose(2,0,1)
        self.sig_xy = sig_xy

        self.runscaling = runscaling
        # Not run when loading class from dictionary
        if runscaling:
            # Real space parameters
            self.x = x.copy()
            self.y = y.copy()
            self.rngx, self.rngy = rngx, rngy
            # Statistics for feature scaling
            self.mux, self.muy, self.sx, self.sy = scales
            # Scaled parameters
            self.x_s, self.y_s = feature_scaling(x, y, self.mux, self.muy, self.sx, self.sy)
            self.rngx_s, self.rngy_s = feature_scaling(np.array(rngx), np.array(rngy), self.mux, self.muy, self.sx, self.sy)
            self.sig_xy_s = covariance_scaling(self.sig_xy, self.sx, self.sy)
        else:
            # Real space parameters
            self.x = x.copy()
            self.y = y.copy()
            self.rngx, self.rngy = rngx, rngy
            self.x_s, self.y_s = x, y
            self.rngx_s, self.rngy_s = rngx, rngy
            self.sig_xy_s = sig_xy

        # Function which calculates the actual distribution
        self.distribution = bivGaussMixture

        # Print out likelihood values as calculated
        self.runningL = runningL

        if runningL: print('N stars = ', len(self.x))

        if self.priorDF:
            # Calculate Gaussian distributions from product of scaled DF and scaled star positions
            if self.runscaling:
                self.params_df = self.scaleParams(self.photoDF.params_f, dfparams=True)
                prior_sfBounds[0,:] = (prior_sfBounds[0,:]-self.mux)/self.sx
                prior_sfBounds[1,:] = (prior_sfBounds[1,:]-self.muy)/self.sy
            else: self.params_df = self.photoDF.params_f
            function = lambda a, b: self.distribution(self.params_df, a, b)
            #if self.runningL:
            #    print 'DF integral = ', numericalIntegrate_precompute(function, self.x_2d, self.y_2d)
            self.ndf = len(self.photoDF.x)
        else: self.ndf = None


    def __call__(self, x, y, components=None, params=None):

        '''
        __call__ - Returns the value of the smooth GMM distribution at the points x, y

        Parameters
        ----------
            x, y - float or np.array of floats
                - x and y coordinates of points at which to take the value of the GMM
                - From input - x is magnitude, y is colour

            components=None:
                - List of components to check for distribution values

            params=None:
                - The parameters on which the model will be evaluatedself.
                - If None, params_f class attribute will be used

        Returns
        -------
            GMMval: float or np.array of floats
                - The value of the GMM at coordinates x, y
        '''
        #
        if params is None: params=self.params_f.copy()

        # Scale x and y to correct region - Currently done to params_f - line 371  - but could change here instead
        #x, y = feature_scaling(x, y, self.mux, self.muy, self.sx, self.sy)
        #rngx, rngy = feature_scaling(np.array(self.rngx), np.array(self.rngy), self.mux, self.muy, self.sx, self.sy)
        rngx, rngy = np.array(self.rngx), np.array(self.rngy)

        # Value of coordinates x, y in the Gaussian mixture model
        if components is None: components = np.arange(self.params_f.shape[0])
        GMMval = self.distribution(params[components, :], x, y)

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

    def optimizeParams(self):

        '''
        optimizeParams - Initialise and optimize parameters of Gaussian mixture model.

        **kwargs
        --------

        Returns
        -------

        '''

        if not self.priorDF:
            # Generate NIW prior parameters
            self.priorDFParams = NIW_prior_params(self.prior_sfBounds, shrinker=7.5)
            params = self.optimize(None, 'BGM')
        if self.priorDF:
            # Generate NIW prior parameters
            self.priorParams = NIW_prior_params(self.prior_sfBounds, shrinker=7.5)
            # Run optimize
            params = self.optimize(self.priorParams, 'TNC_SFonly')

        self.params_f_scaled = params.copy()
        if self.runscaling: params = self.unscaleParams(params)
        # Save evaluated parameters to internal values
        self.params_f = params.copy()

        return params

    def optimize(self, priorParams, method, niter=100):

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

        X = np.vstack((self.x_s, self.y_s)).T

        # To clean up any warnings from optimize
        invalid = np.seterr()['invalid']
        divide = np.seterr()['divide']
        over = np.seterr()['over']
        np.seterr(invalid='ignore', divide='ignore', over='ignore')

        if method=='TNC':
            Xdf = feature_scaling(self.photoDF.x, self.photoDF.y, self.mux, self.muy, self.sx, self.sy)
            Xdf = np.vstack((Xdf[0], Xdf[1])).T
            raw_params = TNC_sf(X, self.priorParams, self.params_df, stdout=self.runningL,
                                Xdf=Xdf, init='best', max_components=self.params_df.shape[0])
            params = transform_sfparams_logit(raw_params)
        elif method=='TNC_HBM':
            Xdf = feature_scaling(self.photoDF.x, self.photoDF.y, self.mux, self.muy, self.sx, self.sy)
            Xdf = np.vstack((Xdf[0], Xdf[1])).T
            gmm = self.photoDF.gmm_df
            df_posterior = [Xdf.shape[0], gmm.weight_concentration_,
                            gmm.means_, gmm.mean_precision_,
                            gmm.covariances_*(gmm.degrees_of_freedom_[:,np.newaxis,np.newaxis]-3),
                            gmm.degrees_of_freedom_]
            raw_params, new_df_params = TNC_sf_HBM(X, self.priorParams, self.params_df, df_posterior, stdout=self.runningL,
                                            Xdf=Xdf, init='best', max_components=self.params_df.shape[0])
            params = transform_sfparams_logit(raw_params)
            self.new_df_params = new_df_params
        elif method=='TNC_SFonly':
            Xdf = feature_scaling(self.photoDF.x, self.photoDF.y, self.mux, self.muy, self.sx, self.sy)
            Xdf = np.vstack((Xdf[0], Xdf[1])).T
            raw_params = TNC_sf_SFonly(X, self.priorParams, self.params_df, stdout=self.runningL,
                                            Xdf=Xdf, init='best', max_components=self.params_df.shape[0])
            params = transform_sfparams_logit(raw_params)
        elif method=='BGM':
            print('Running BGM')
            params, gmm = BGMM_df(X, self.priorDFParams, stdout=self.runningL)
            self.gmm_df = gmm
        elif method=='emcee':
            params = transform_sfparams_invlogit(self.params_f_scaled)
            self.sampler = BGMM_emcee_ball(params, X, self.priorParams, self.params_df, niter=niter)
            return self.sampler
        elif method=='emceeBHM':
            Xdf = feature_scaling(self.photoDF.x, self.photoDF.y, self.mux, self.muy, self.sx, self.sy)
            Xdf = np.vstack((Xdf[0], Xdf[1])).T
            df_params = self.params_df.copy()
            params = transform_sfparams_invlogit(self.params_f_scaled)
            p0 = np.hstack((params.flatten(), df_params.flatten(), np.array([len(self.photoDF.x)])))
            gmm = self.photoDF.gmm_df
            df_posterior = [Xdf.shape[0], gmm.weight_concentration_,
                            gmm.means_, gmm.mean_precision_,
                            gmm.covariances_*(gmm.degrees_of_freedom_[:,np.newaxis,np.newaxis]-3),
                            gmm.degrees_of_freedom_]
            params = np.vstack((params, df_params))
            self.sampler = BGMM_emcee_ball_BHM(params, X, self.priorParams, df_posterior, niter=niter)
            return self.sampler
        elif method=='emceeSFonly':
            params = transform_sfparams_invlogit(self.params_f_scaled)
            self.sampler = BGMM_emcee_ball_SFonly(params, X, self.priorParams, self.params_df, niter=niter)
            return self.sampler
        else:
            raise ValueError('What is the method???')


        # To clean up any warnings from optimize
        np.seterr(invalid=invalid, divide=divide, over=over)
        if self.runningL:
            print('Param shape: ', params.shape)
            print("")

        return params

    def scaleParams(self, params_in, dfparams=False):

        # This isn't quite right, the likelihood doesn't turn out the same!

        params = params_in.copy()
        params[:,[0,1]] = (params[:,[0,1]] -  [self.mux, self.muy]) / np.array([self.sx, self.sy])

        params[:,[2,3]] *= 1/np.array([self.sx**2, self.sy**2])

        #if self.priorDF & (not dfparams):
        #    params[:,5] /= (self.sx*self.sy)
        params[:,5] /= (self.sx*self.sy)

        return params

    def unscaleParams(self, params_in, dfparams=False):

        params = params_in.copy()
        params[:,[0,1]] = (params[:,[0,1]] * np.array([self.sx, self.sy])) +  [self.mux, self.muy]

        params[:,[2,3]] *= np.array([self.sx**2, self.sy**2])

        #if self.priorDF & (not dfparams):
        params[:,5] *= (self.sx*self.sy)

        return params


# General functions
def quick_invdet(S):
    det = S[:,0,0]*S[:,1,1] - S[:,0,1]**2
    Sinv = S.copy()*0
    Sinv[:,0,0] = S[:,1,1]
    Sinv[:,1,1] = S[:,0,0]
    Sinv[:,0,1] = -S[:,0,1]
    Sinv[:,1,0] = -S[:,1,0]
    Sinv *= 1/det[:,np.newaxis,np.newaxis]

    return Sinv, det
def Gaussian_i(delta, Sinv, Sdet):
    # delta is [nStar, nComponent, 2]
    # Sinv is [nComponent, 2, 2]
    # Sdet is [nComponent]

    sum_axis = len(delta.shape)-1
    exponent = -0.5*np.sum(delta * \
                    np.sum(Sinv[np.newaxis,...]*delta[...,np.newaxis], axis=sum_axis), axis=sum_axis)
    norm = 1/(2*np.pi*np.sqrt(Sdet))

    # Return shape is (Nstar, Ncomponent)
    return norm[np.newaxis] * np.exp(exponent)
def Gaussian_int(delta, Sinv, Sdet):
    # delta is [nComponent, 2]
    # Sinv is [nComponent, 2, 2]
    # Sdet is [nComponent]

    sum_axis = len(delta.shape)-1
    exponent = -0.5*np.sum(delta * \
                    np.sum(Sinv*delta[...,np.newaxis], axis=sum_axis), axis=sum_axis)
    norm = 1/(2*np.pi*np.sqrt(Sdet))

    # Return shape is (Ncomponent)
    return norm * np.exp(exponent)

# Manipulating parameters
def NIW_prior_params(bounds, l0=1e-4, nu0=2., shrinker=5):

    mu0 = (bounds[:,1] + bounds[:,0])/2
    std0 = (bounds[:,1] - bounds[:,0])/2
    Psi0 = np.array([[std0[0]**2, 0.], [0., std0[1]**2]])/(shrinker**2)
    #mu0_scaled = np.mean(Xsf, axis=0)
    #Psi0 = np.mean((Xsf-mu0)[...,np.newaxis] * (Xsf-mu0)[...,np.newaxis,:], axis=0)
    priorParams = [mu0, l0, Psi0, nu0]
    print('priorParams: ', priorParams)

    return priorParams
def get_params(gmm_inst, Nstar, n_components):

    params = np.zeros((n_components,6))
    params[:,:2] = gmm_inst.means_
    params[:,2:4] = gmm_inst.covariances_[:,[0,1],[0,1]]
    params[:,4] = gmm_inst.covariances_[:,0,1]/np.sqrt(gmm_inst.covariances_[:,0,0]*gmm_inst.covariances_[:,1,1])
    params[:,5] = gmm_inst.weights_*Nstar

    return params
def get_sfparams_logit(gmm_inst, n_components):

    params = np.zeros((n_components,6))
    params[:,:2] = gmm_inst.means_
    params[:,2:4] = gmm_inst.covariances_[:,[0,1],[0,1]]
    params[:,4] = gmm_inst.covariances_[:,0,1]/np.sqrt(gmm_inst.covariances_[:,0,0]*gmm_inst.covariances_[:,1,1])
    params[:,5] = gmm_inst.weights_

    # Constrain the distribution maxima
    sigma = np.array([[params[:,2], np.sqrt(params[:,2]*params[:,3])*params[:,4]],
                      [np.sqrt(params[:,2]*params[:,3])*params[:,4], params[:,3]]])
    sigma = np.moveaxis(sigma, -1, 0)
    sigma_inv, sigma_det = quick_invdet(sigma)
    norm = 1/(2*np.pi*np.sqrt(sigma_det))
    maxima = gradient_rootfinder(params, sigma_inv, norm)
    if maxima>=1:
        params[:,5] *= 0.9/maxima

    # logit
    params[:,4] = (params[:,4]+1)/2.
    params[:,4] = np.log(params[:,4]/(1-params[:,4]))

    w = params[:,5]
    w *= norm
    w[w>1] = 0.9
    params[:,5] = np.log(w/(1-w))
    # Logit on correlation

    return params
def transform_sfparams_logit(params):

    raw_params = params.copy().reshape(-1,6)
    #print(raw_params[:,5])

    raw_params[:,2:4] = np.abs(raw_params[:,2:4])

    e_alpha = np.exp(-raw_params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    cov = np.sqrt(raw_params[...,2]*raw_params[...,3])*(2*p - 1)
    S_sf = np.moveaxis(np.array([[raw_params[...,2], cov], [cov, raw_params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)
    raw_params[...,4] = (2*p - 1)

    # Logit correction of raw_params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-raw_params[...,5])
    pi_scaled = 1./(1.+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * pi_scaled
    raw_params[:,5] = pi

    return raw_params
def transform_sfparams_invlogit(raw_params):

    params = raw_params.copy().reshape(-1,6)

    # logit scaled correlation
    p = (params[:,4]+1)/2
    params[:,4] = np.log(p/(1-p))

    cov = raw_params[:,4]*np.sqrt(raw_params[:,2]*raw_params[:,3])
    S_sf = np.moveaxis(np.array([[raw_params[...,2], cov], [cov, raw_params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)

    # Logit correction of raw_params[:,5] - [0, rt(det(2.pi.S))] --> [-inf, inf]
    pi = raw_params[:,5].copy()
    pi_scaled = pi / (2*np.pi*np.sqrt(Sdet_sf))
    pi_scaled[pi_scaled>=1] = (1-1e-10)
    params[:,5] = np.log(pi_scaled/(1-pi_scaled))

    return params
def transform_dfparams_logit(params):

    raw_params = params.copy().reshape(-1,6)
    #print(raw_params[:,5])
    raw_params[:,2:4] = np.abs(raw_params[:,2:4])

    e_alpha = np.exp(-raw_params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    raw_params[...,4] = (2*p - 1)

    # Logit correction of raw_params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-raw_params[...,5])
    pi = 1./(1.+e_alpha_pi)
    raw_params[:,5] = pi

    return raw_params
def transform_dfparams_invlogit(raw_params):

    params = raw_params.copy().reshape(-1,6)

    # logit scaled correlation
    p = (params[:,4]+1)/2
    params[:,4] = np.log(p/(1-p))

    cov = raw_params[:,4]*np.sqrt(raw_params[:,2]*raw_params[:,3])
    S_sf = np.moveaxis(np.array([[raw_params[...,2], cov], [cov, raw_params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)

    # Logit correction of raw_params[:,5] - [0, rt(det(2.pi.S))] --> [-inf, inf]
    pi = raw_params[:,5].copy()
    params[:,5] = np.log(pi/(1-pi))

    return params
def gmm_product_params(params1, params2):

    idx1 = np.repeat(np.arange(params1.shape[0]), params2.shape[0])
    idx2 = np.tile(np.arange(params2.shape[0]), params1.shape[0])

    m1 = params1[:,:2]
    m2 = params2[:,:2]

    # Covariance matrices
    cov1 = np.sqrt(params1[...,2]*params1[...,3])*params1[...,4]
    S1 = np.moveaxis(np.array([[params1[...,2], cov1], [cov1, params1[...,3]]]), -1, 0)
    cov2 = np.sqrt(params2[...,2]*params2[...,3])*params2[...,4]
    S2 = np.moveaxis(np.array([[params2[...,2], cov2], [cov2, params2[...,3]]]), -1, 0)

    S1_inv, S1_det = quick_invdet(S1)
    S2_inv, S2_det = quick_invdet(S2)
    S3 = quick_invdet(S1_inv[idx1]+S2_inv[idx2])[0]

    m3 = np.sum(S3 * np.sum((S1_inv*m1[:,:,np.newaxis])[idx1] + (S2_inv*m2[:,:,np.newaxis])[idx2], axis=1)[:,:,np.newaxis], axis=1)

    delta_mm = m1[idx1] - m2[idx2]
    S1_S2_inv, S1_S2_det = quick_invdet(S1[idx1]+S2[idx2])
    exponent = -0.5 * np.sum(delta_mm * np.sum(S1_S2_inv * delta_mm[:,:,np.newaxis], axis=1), axis=1)
    norm = 1/(2*np.pi*np.sqrt(S1_S2_det))
    cc = norm*np.exp(exponent)

    weights = params1[:,5][idx1]*params2[:,5][idx2]

    params3 = np.zeros((S3.shape[0], 6))
    params3[:,:2] = m3
    params3[:,2:4] = S3[:,[0,1],[0,1]]
    params3[:,4] = S3[:,0,1]/(np.sqrt(S3[:,0,0]*S3[:,1,1]))
    params3[:,5] = weights*cc

    return params3


def gmm_gradient(X, sigma_inv,  mu, weight, norm):

    delta = mu-X[np.newaxis,:]
    sigma_inv_delta = np.sum(sigma_inv*delta[:,:,np.newaxis], axis=1)

    # Gaussian
    exponent = np.exp(-0.5 * np.sum(delta*sigma_inv_delta, axis=1))
    fx = norm*exponent

    # Jacobian
    jac = -sigma_inv_delta * fx[:,np.newaxis]

    result =  -np.sum(jac*weight[:,np.newaxis], axis=0)
    #print(X.shape, result.shape)
    return result
def gradient_rootfinder(params, sigma_inv, norm, all_maxima=False):

    loc = np.zeros((params.shape[0], 2))
    for i in range(params.shape[0]):
        opt = op.root(gmm_gradient, params[i,:2],
                                      args=(sigma_inv, params[:,:2], params[:,5], norm))
        loc[i] = opt.x

    #if not np.product(~np.isnan(norm)): print(params[:,0])

    out = bivGaussMixture(params, loc[:,0], loc[:,1])

    if all_maxima:
        return out, loc
    return np.max(out)
    #return out, loc

# Likelihood, Prior and Posterior functions
def calc_nlnP_pilogit_NIW(params, Xsf, NIWprior, df_params, stdout=False):

    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    # means
    params[:,2:4] = np.abs(params[:,2:4])
    df_idx = np.repeat(np.arange(df_params.shape[0]), params.shape[0])
    sf_idx = np.tile(np.arange(params.shape[0]), df_params.shape[0])
    # covariances
    e_alpha = np.exp(-params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr = (2*p - 1)
    cov = np.sqrt(params[...,2]*params[...,3])*corr
    S_sf = np.moveaxis(np.array([[params[...,2], cov], [cov, params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)
    delta_sf = Xsf[:,np.newaxis,:]-params[:,:2][np.newaxis,:,:]
    Sinv_sf_delta = np.sum(Sinv_sf[np.newaxis,...]*delta_sf[...,np.newaxis], axis=2)
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-params[...,5])
    p_pi = 1./(1+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * p_pi

    # Unscaled parameters
    params_original = params.copy()
    params_original[:,4] = corr
    params_original[:,5] = pi
    # Max SF prior
    gmm_maxima = gradient_rootfinder(params_original, Sinv_sf, 1/(2*np.pi*np.sqrt(Sdet_sf)))
    #print(gmm_maxima)
    if gmm_maxima>1:
        return 1e10

    # Likelihood
    corr = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], corr], [corr, df_params[...,3]]]), -1, 0)

    Sinv_sum, Sdet_sum = quick_invdet(S_sf[sf_idx]+S_df[df_idx])
    Sinv_sum_delta = np.sum(Sinv_sum*(df_params[:,:2][df_idx] - params[:,:2][sf_idx])[...,np.newaxis], axis=1)

    # Star iteration term
    # (Nstar x Ncomponent_sf)
    m_ij = Gaussian_i(delta_sf, Sinv_sf, Sdet_sf)
    m_i = np.sum(pi*m_ij, axis=1)

    # Integral Term
    delta_mumu = params[:,:2][sf_idx] - df_params[:,:2][df_idx]
    I_jl = Gaussian_int(delta_mumu, Sinv_sum, Sdet_sum)  * pi[sf_idx] * df_params[:,5][df_idx]
    I = np.sum(I_jl)

    # Prior
    m0, l0, Psi0, nu0 = NIWprior
    delta_mumu0 = params[:,:2]-m0[np.newaxis,:]
    Prior0 = (-(nu0+4.)/2.) * np.log(Sdet_sf)
    Priormu = (-l0/2.) * np.sum(delta_mumu0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1), axis=1)
    PriorS = (-1/2.) * np.trace(np.matmul(Psi0[np.newaxis,...], Sinv_sf), axis1=-2, axis2=-1)
    Prior = np.sum(Prior0 + Priormu + PriorS)

    nlnP = - ( np.sum(np.log(m_i)) - np.sum(I) + Prior)
    #print('%.0f' % nlnP, '.....', end='')
    return  nlnP
def calc_nlnP_grad_pilogit_NIW(params, Xsf, NIWprior, df_params, stdout=False):

    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    # means
    params[:,2:4] = np.abs(params[:,2:4])
    df_idx = np.repeat(np.arange(df_params.shape[0]), params.shape[0])
    sf_idx = np.tile(np.arange(params.shape[0]), df_params.shape[0])
    # covariances
    e_alpha = np.exp(-params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr = (2*p - 1)
    cov = np.sqrt(params[...,2]*params[...,3])*corr
    S_sf = np.moveaxis(np.array([[params[...,2], cov], [cov, params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)
    delta_sf = Xsf[:,np.newaxis,:]-params[:,:2][np.newaxis,:,:]
    Sinv_sf_delta = np.sum(Sinv_sf[np.newaxis,...]*delta_sf[...,np.newaxis], axis=2)
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-params[...,5])
    p_pi = 1./(1+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * p_pi

    # Unscaled parameters
    params_original = params.copy()
    params_original[:,4] = corr
    params_original[:,5] = pi
    # Max SF prior
    gmm_maxima = gradient_rootfinder(params_original, Sinv_sf, 1/(2*np.pi*np.sqrt(Sdet_sf)))
    if gmm_maxima>1:
        return 1e100, params.flatten().copy()*0.#grad.flatten()

    # Likelihood
    corr = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], corr], [corr, df_params[...,3]]]), -1, 0)

    Sinv_sum, Sdet_sum = quick_invdet(S_sf[sf_idx]+S_df[df_idx])
    Sinv_sum_delta = np.sum(Sinv_sum*(df_params[:,:2][df_idx] - params[:,:2][sf_idx])[...,np.newaxis], axis=1)

    # Star iteration term
    # (Nstar x Ncomponent_sf)
    m_ij = Gaussian_i(delta_sf, Sinv_sf, Sdet_sf)
    m_i = np.sum(pi*m_ij, axis=1)

    # Integral Term
    delta_mumu = params[:,:2][sf_idx] - df_params[:,:2][df_idx]
    I_jl = Gaussian_int(delta_mumu, Sinv_sum, Sdet_sum)  * pi[sf_idx] * df_params[:,5][df_idx]
    I = np.sum(I_jl)

    # Prior
    m0, l0, Psi0, nu0 = NIWprior
    delta_mumu0 = params[:,:2]-m0[np.newaxis,:]
    Prior0 = (-(nu0+4.)/2.) * np.log(Sdet_sf)
    Priormu = (-l0/2.) * np.sum(delta_mumu0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1), axis=1)
    PriorS = (-1/2.) * np.trace(np.matmul(Psi0[np.newaxis,...], Sinv_sf), axis1=-2, axis2=-1)
    Prior = np.sum(Prior0 + Priormu + PriorS)
    # Calculation for later
    Sinv_delta_mumu0 = np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1)

    # Gradients

    # Pi
    A = np.sum(m_ij/m_i[:,np.newaxis], axis=0) # i-term
    B = np.sum((I_jl/pi[sf_idx]).reshape(df_params.shape[0], params.shape[0]), axis=0) # int-term
    C = 1/pi # prior-term
    gradPi = A - B

    # mu
    A = (pi[np.newaxis,:]*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis]*\
        Sinv_sf_delta
    A = np.sum(A, axis=0) # i-term (nComponent x 2)
    B = (I_jl)[:,np.newaxis] * \
        Sinv_sum_delta
    B = np.sum(B.reshape(df_params.shape[0], params.shape[0], 2), axis=0) # int-term
    C = -l0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1) # NIW prior
    gradmu = A - B + C

    #sigma
    diff = -0.5*(Sinv_sf[np.newaxis] - Sinv_sf_delta[...,np.newaxis]*Sinv_sf_delta[...,np.newaxis,:])
    A = (pi*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis,np.newaxis] * diff
    A = np.sum(A, axis=0) # i-term (nComponent x 2 x 2)
    diff = -0.5*(Sinv_sum - Sinv_sum_delta[...,np.newaxis]*Sinv_sum_delta[...,np.newaxis,:])
    B = (I_jl)[:,np.newaxis,np.newaxis] * diff
    B = np.sum(B.reshape(df_params.shape[0], params.shape[0], 2, 2), axis=0) # int-term (nComponent x 2 x 2)
    C = -((nu0+4)/2.)*Sinv_sf - (l0/2.)*Sinv_delta_mumu0[...,np.newaxis]*Sinv_delta_mumu0[...,np.newaxis,:]\
        + (1./2.) * np.matmul(Sinv_sf, np.matmul(Psi0[np.newaxis,...], Sinv_sf))
    gradS = A - B + C

    grad = np.zeros((params.shape[0],6))
    grad[:,:2] = gradmu
    grad[:,2] = gradS[:,0,0]
    grad[:,3] = gradS[:,1,1]
    grad[:,4] = 2*gradS[:,0,1]*np.sqrt(params[:,2]*params[:,3])*2*p**2*e_alpha
    grad[:,5] = gradPi * 2*np.pi*np.sqrt(Sdet_sf) * p_pi**2 * e_alpha_pi

    return  - ( np.sum(np.log(m_i)) - np.sum(I) + Prior), -grad.flatten()
def calc_nlnP_pilogit_NIW_DFfit(params, Xsf, NIWprior, Post_df, stdout=False):

    # Prior on distribution function components
    Nphot, conc_df, mdf, ldf, Psidf, nudf = Post_df
    ncomponents_df = mdf.shape[0]

    # Parameters - transform to means, covariances and weights
    N = params[-1]
    params = np.reshape(params[:-1], (-1,6))
    params[:,2:4] = np.abs(params[:,2:4])
    df_params = params[-ncomponents_df:].copy()
    params = params[:-ncomponents_df].copy()
    # means
    df_idx = np.repeat(np.arange(df_params.shape[0]), params.shape[0])
    sf_idx = np.tile(np.arange(params.shape[0]), df_params.shape[0])
    # covariances
    e_alpha = np.exp(-params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr_sf = (2*p - 1)
    cov = np.sqrt(params[...,2]*params[...,3])*corr_sf
    S_sf = np.moveaxis(np.array([[params[...,2], cov], [cov, params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)
    delta_sf = Xsf[:,np.newaxis,:]-params[:,:2][np.newaxis,:,:]
    Sinv_sf_delta = np.sum(Sinv_sf[np.newaxis,...]*delta_sf[...,np.newaxis], axis=2)
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-params[...,5])
    p_pi = 1./(1+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * p_pi

    # Nor for the DF
    # covariances
    e_alpha = np.exp(-df_params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr = (2*p - 1)
    cov = np.sqrt(df_params[...,2]*df_params[...,3])*corr
    S_df = np.moveaxis(np.array([[df_params[...,2], cov], [cov, df_params[...,3]]]), -1, 0)
    Sinv_df, Sdet_df = quick_invdet(S_df)
    delta_df = Xsf[:,np.newaxis,:]-df_params[:,:2][np.newaxis,:,:]
    Sinv_df_delta = np.sum(Sinv_df[np.newaxis,...]*delta_df[...,np.newaxis], axis=2)
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-df_params[...,5])
    pi_df = 1./(1+e_alpha_pi)

    # Unscaled parameters
    params_original = params.copy()
    params_original[:,4] = corr_sf
    params_original[:,5] = pi
    # Max SF prior
    gmm_maxima = gradient_rootfinder(params_original, Sinv_sf, 1/(2*np.pi*np.sqrt(Sdet_sf)))
    #print(gmm_maxima)
    if gmm_maxima>1:
        return 1e10

    # Joint sf-df model
    Sinv_sum, Sdet_sum = quick_invdet(S_sf[sf_idx]+S_df[df_idx])
    Sinv_sum_delta = np.sum(Sinv_sum*(df_params[:,:2][df_idx] - params[:,:2][sf_idx])[...,np.newaxis], axis=1)

    # Likelihood
    # SF star iteration term (Nstar x Ncomponent_sf)
    m_ij = Gaussian_i(delta_sf, Sinv_sf, Sdet_sf)
    m_i = np.sum(pi*m_ij, axis=1)
    # DF star iteration term (Nstar x Ncomponent_sf)
    m_ij_df = Gaussian_i(delta_df, Sinv_df, Sdet_df)
    m_i_df = np.sum(pi_df*m_ij_df, axis=1)
    # Integral Term
    delta_mumu = params[:,:2][sf_idx] - df_params[:,:2][df_idx]
    I_jl = Gaussian_int(delta_mumu, Sinv_sum, Sdet_sum)  * pi[sf_idx] * df_params[:,5][df_idx]
    I = np.sum(I_jl)

    # Prior
    m0, l0, Psi0, nu0 = NIWprior
    delta_mumu0 = params[:,:2]-m0[np.newaxis,:]
    Prior0 = (-(nu0+4.)/2.) * np.log(Sdet_sf)
    Priormu = (-l0/2.) * np.sum(delta_mumu0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1), axis=1)
    PriorS = (-1/2.) * np.trace(np.matmul(Psi0[np.newaxis,...], Sinv_sf), axis1=-2, axis2=-1)
    Prior = np.sum(Prior0 + Priormu + PriorS)

    delta_mumudf = df_params[:,:2]-mdf
    PriorN_df = N*np.log(Nphot/N) - (Nphot-N)
    PriorPi_df = (conc_df - 1) * np.log(pi_df)
    Prior0_df = (-(nudf+4.)/2.) * np.log(Sdet_df)
    Priormu_df = (-ldf/2.) * np.sum(delta_mumudf * np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1), axis=1)
    PriorS_df = (-1/2.) * np.trace(np.matmul(Psidf, Sinv_df), axis1=-2, axis2=-1)
    Prior_df = np.sum(PriorN_df + PriorPi_df + Prior0_df + Priormu_df + PriorS_df)

    nlnP = - ( np.log(N) + np.sum(np.log(m_i_df)) + np.sum(np.log(m_i)) - np.sum(I) + Prior + Prior_df)
    #print('%.0f' % nlnP, '.....', end='')


    if np.isnan(nlnP):
        print('Nan Prob')
        print(np.log(N), np.sum(np.log(m_i_df)), np.sum(np.log(m_i)), np.sum(I), Prior, Prior_df)
        print('DF prior components: ', PriorN_df, PriorPi_df, Prior0_df, Priormu_df, PriorS_df)
        print('Pi prior components: ', conc_df, pi_df)

    return  nlnP
def calc_prior_df(params, Xsf, NIWprior, Post_df, stdout=False):
    # Prior on distribution function components
    Nphot, conc_df, mdf, ldf, Psidf, nudf = Post_df
    ncomponents_df = mdf.shape[0]

    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    params[:,2:4] = np.abs(params[:,2:4])
    df_params = params[-ncomponents_df:].copy()
    N = np.sum(df_params[:,5])
    # Prior on df_params
    if (np.sum(np.abs(df_params[...,4])>=1)>0)|(np.sum(df_params[:,5]<0)>0)|(N<0):
        return 1e10

    # Nor for the DF
    # covariances
    cov = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], cov], [cov, df_params[...,3]]]), -1, 0)
    Sinv_df, Sdet_df = quick_invdet(S_df)
    delta_df = Xsf[:,np.newaxis,:]-df_params[:,:2][np.newaxis,:,:]
    Sinv_df_delta = np.sum(Sinv_df[np.newaxis,...]*delta_df[...,np.newaxis], axis=2)
    #weights
    pi_df = df_params[:,5].copy()/N

    delta_mumudf = df_params[:,:2]-mdf
    PriorN_df = N*np.log(Nphot/N) - (Nphot-N)
    PriorPi_df = (conc_df - 1) * np.log(pi_df)
    Prior0_df = (-(nudf+4.)/2.) * np.log(Sdet_df)
    Priormu_df = (-ldf/2.) * np.sum(delta_mumudf * np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1), axis=1)
    PriorS_df = (-1/2.) * np.trace(np.matmul(Psidf, Sinv_df), axis1=-2, axis2=-1)
    Prior_df = np.sum(PriorN_df + PriorPi_df + Prior0_df + Priormu_df + PriorS_df)

    nlnP = - Prior_df
    #print('%.0f' % nlnP, '.....', end='')

    return  nlnP

def calc_nlnP_NIW_DFfit(params, Xsf, NIWprior, Post_df, stdout=False):

    # Prior on distribution function components
    Nphot, conc_df, mdf, ldf, Psidf, nudf = Post_df
    ncomponents_df = mdf.shape[0]

    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    params[:,2:4] = np.abs(params[:,2:4])
    df_params = params[-ncomponents_df:].copy()
    params = params[:-ncomponents_df].copy()
    N = np.sum(df_params[:,5])
    # Prior on df_params
    if (np.sum(np.abs(df_params[...,4])>=1)>0)|(np.sum(df_params[:,5]<0)>0)|(N<0):
        return 1e10
    # means
    df_idx = np.repeat(np.arange(df_params.shape[0]), params.shape[0])
    sf_idx = np.tile(np.arange(params.shape[0]), df_params.shape[0])
    # covariances
    e_alpha = np.exp(-params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr_sf = (2*p - 1)
    cov = np.sqrt(params[...,2]*params[...,3])*corr_sf
    S_sf = np.moveaxis(np.array([[params[...,2], cov], [cov, params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)
    delta_sf = Xsf[:,np.newaxis,:]-params[:,:2][np.newaxis,:,:]
    Sinv_sf_delta = np.sum(Sinv_sf[np.newaxis,...]*delta_sf[...,np.newaxis], axis=2)
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-params[...,5])
    p_pi = 1./(1+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * p_pi

    # Nor for the DF
    # covariances
    cov = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], cov], [cov, df_params[...,3]]]), -1, 0)
    Sinv_df, Sdet_df = quick_invdet(S_df)
    delta_df = Xsf[:,np.newaxis,:]-df_params[:,:2][np.newaxis,:,:]
    Sinv_df_delta = np.sum(Sinv_df[np.newaxis,...]*delta_df[...,np.newaxis], axis=2)
    #weights
    pi_df = df_params[:,5].copy()/N

    # Unscaled parameters
    params_original = params.copy()
    params_original[:,4] = corr_sf
    params_original[:,5] = pi
    # Max SF prior
    gmm_maxima = gradient_rootfinder(params_original, Sinv_sf, 1/(2*np.pi*np.sqrt(Sdet_sf)))
    #print(gmm_maxima)
    if gmm_maxima>1:
        return 1e10

    # Joint sf-df model
    Sinv_sum, Sdet_sum = quick_invdet(S_sf[sf_idx]+S_df[df_idx])
    Sinv_sum_delta = np.sum(Sinv_sum*(df_params[:,:2][df_idx] - params[:,:2][sf_idx])[...,np.newaxis], axis=1)

    # Likelihood
    # SF star iteration term (Nstar x Ncomponent_sf)
    m_ij = Gaussian_i(delta_sf, Sinv_sf, Sdet_sf)
    m_i = np.sum(pi*m_ij, axis=1)
    # DF star iteration term (Nstar x Ncomponent_sf)
    m_ij_df = Gaussian_i(delta_df, Sinv_df, Sdet_df)
    m_i_df = np.sum(pi_df*m_ij_df, axis=1)
    # Integral Term
    delta_mumu = params[:,:2][sf_idx] - df_params[:,:2][df_idx]
    I_jl = Gaussian_int(delta_mumu, Sinv_sum, Sdet_sum)  * pi[sf_idx] * df_params[:,5][df_idx]
    I = np.sum(I_jl)

    # Prior
    m0, l0, Psi0, nu0 = NIWprior
    delta_mumu0 = params[:,:2]-m0[np.newaxis,:]
    Prior0 = (-(nu0+4.)/2.) * np.log(Sdet_sf)
    Priormu = (-l0/2.) * np.sum(delta_mumu0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1), axis=1)
    PriorS = (-1/2.) * np.trace(np.matmul(Psi0[np.newaxis,...], Sinv_sf), axis1=-2, axis2=-1)
    Prior = np.sum(Prior0 + Priormu + PriorS)

    delta_mumudf = df_params[:,:2]-mdf
    PriorN_df = N*np.log(Nphot/N) - (Nphot-N)
    PriorPi_df = (conc_df - 1) * np.log(pi_df)
    Prior0_df = (-(nudf+4.)/2.) * np.log(Sdet_df)
    Priormu_df = (-ldf/2.) * np.sum(delta_mumudf * np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1), axis=1)
    PriorS_df = (-1/2.) * np.trace(np.matmul(Psidf, Sinv_df), axis1=-2, axis2=-1)
    Prior_df = np.sum(PriorN_df + PriorPi_df + Prior0_df + Priormu_df + PriorS_df)

    nlnP = - ( np.log(N) + np.sum(np.log(m_i_df)) + np.sum(np.log(m_i)) - np.sum(I) + Prior + Prior_df)
    #print('%.0f' % nlnP, '.....', end='')


    if np.isnan(nlnP):
        print('Nan Prob')
        print(np.log(N), np.sum(np.log(m_i_df)), np.sum(np.log(m_i)), np.sum(I), Prior, Prior_df)
        print('DF prior components: ', PriorN_df, PriorPi_df, Prior0_df, Priormu_df, PriorS_df)
        print('Pi prior components: ', conc_df, pi_df)
        print('Constraints: ', np.abs(df_params[...,4])>=1, df_params[:,5]<0, N<0)
        print('Constraints sum: ', np.sum(np.abs(df_params[...,4])>=1), np.sum(df_params[:,5]<0), N<0)
        print('Constraints bool: ', np.sum(np.abs(df_params[...,4])>=1)>0, np.sum(df_params[:,5]<0)>0, N<0)
        print('Constraints combined: ', (np.sum(np.abs(df_params[...,4])>=1)>0)|(np.sum(df_params[:,5]<0)>0)|(N<0))

    return  nlnP
def calc_nlnP_grad_pilogit_NIW_DFfit(params, Xsf, NIWprior, Post_df, stdout=False):

    # Prior on distribution function components
    Nphot, conc_df, mdf, ldf, Psidf, nudf = Post_df
    ncomponents_df = mdf.shape[0]

    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    params[:,2:4] = np.abs(params[:,2:4])
    df_params = params[-ncomponents_df:].copy()
    params = params[:-ncomponents_df].copy()
    N = np.sum(df_params[:,5])
    # Prior on df_params
    if (np.sum(np.abs(df_params[...,4])>=1)>0)|(np.sum(df_params[:,5]<0)>0)|(N<0):
        return 1e10
    # means
    df_idx = np.repeat(np.arange(df_params.shape[0]), params.shape[0])
    sf_idx = np.tile(np.arange(params.shape[0]), df_params.shape[0])
    # covariances
    e_alpha = np.exp(-params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr = (2*p - 1)
    cov = np.sqrt(params[...,2]*params[...,3])*corr
    S_sf = np.moveaxis(np.array([[params[...,2], cov], [cov, params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf)
    delta_sf = Xsf[:,np.newaxis,:]-params[:,:2][np.newaxis,:,:]
    Sinv_sf_delta = np.sum(Sinv_sf[np.newaxis,...]*delta_sf[...,np.newaxis], axis=2)
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-params[...,5])
    p_pi = 1./(1+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * p_pi

    # Now for the DF
    cov = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], cov], [cov, df_params[...,3]]]), -1, 0)
    Sinv_df, Sdet_df = quick_invdet(S_df)
    delta_df = Xsf[:,np.newaxis,:]-df_params[:,:2][np.newaxis,:,:]
    Sinv_df_delta = np.sum(Sinv_df[np.newaxis,...]*delta_df[...,np.newaxis], axis=2)
    #weights
    pi_df = df_params[:,5].copy()/N

    # Unscaled parameters
    params_original = params.copy()
    params_original[:,4] = corr
    params_original[:,5] = pi
    # Max SF prior
    gmm_maxima = gradient_rootfinder(params_original, Sinv_sf, 1/(2*np.pi*np.sqrt(Sdet_sf)))
    #print(gmm_maxima)
    if gmm_maxima>1:
        return 1e10

    # Joint sf-df model
    Sinv_sum, Sdet_sum = quick_invdet(S_sf[sf_idx]+S_df[df_idx])
    Sinv_sum_delta = np.sum(Sinv_sum*(df_params[:,:2][df_idx] - params[:,:2][sf_idx])[...,np.newaxis], axis=1)

    # Likelihood
    # SF star iteration term (Nstar x Ncomponent_sf)
    m_ij = Gaussian_i(delta_sf, Sinv_sf, Sdet_sf)
    m_i = np.sum(pi*m_ij, axis=1)
    # DF star iteration term (Nstar x Ncomponent_sf)
    m_ij_df = Gaussian_i(delta_df, Sinv_df, Sdet_df)
    m_i_df = np.sum(pi_df*m_ij_df, axis=1)
    # Integral Term
    delta_mumu = params[:,:2][sf_idx] - df_params[:,:2][df_idx]
    I_jl = Gaussian_int(delta_mumu, Sinv_sum, Sdet_sum)  * pi[sf_idx] * df_params[:,5][df_idx]
    I = np.sum(I_jl)

    # Prior
    m0, l0, Psi0, nu0 = NIWprior
    delta_mumu0 = params[:,:2]-m0[np.newaxis,:]
    Prior0 = (-(nu0+4.)/2.) * np.log(Sdet_sf)
    Priormu = (-l0/2.) * np.sum(delta_mumu0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1), axis=1)
    PriorS = (-1/2.) * np.trace(np.matmul(Psi0[np.newaxis,...], Sinv_sf), axis1=-2, axis2=-1)
    Prior = np.sum(Prior0 + Priormu + PriorS)

    delta_mumudf = df_params[:,:2]-mdf
    PriorN_df = N*np.log(Nphot/N) - (Nphot-N)
    PriorPi_df = (conc_df/np.sum(conc_df) - 1) * np.log(pi_df/np.sum(pi_df))
    Prior0_df = (-(nudf+4.)/2.) * np.log(Sdet_df)
    Priormu_df = (-ldf/2.) * np.sum(delta_mumudf * np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1), axis=1)
    PriorS_df = (-1/2.) * np.trace(np.matmul(Psidf, Sinv_df), axis1=-2, axis2=-1)
    Prior_df = np.sum(PriorN_df + PriorPi_df + Prior0_df + Priormu_df + PriorS_df)

    nlnP = - ( np.log(N) + np.sum(np.log(m_i_df)) + np.sum(np.log(m_i)) - np.sum(I) + Prior + Prior_df)
    if stdout:
        print('PriorS: ', PriorS_df)
        print(PriorN_df, np.sum(PriorPi_df), np.sum(Prior0_df), np.sum(Priormu_df), np.sum(PriorS_df))
        print(np.log(N), np.sum(np.log(m_i_df)), np.sum(np.log(m_i)), np.sum(I), Prior, Prior_df)

    # Calculation for later
    Sinv_delta_mumu0 = np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1)
    Sinv_delta_mumudf = np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1)


    # Gradients
    # Pi
    A = np.sum(m_ij/m_i[:,np.newaxis], axis=0) # i-term
    B = np.sum((I_jl/pi[sf_idx]).reshape(df_params.shape[0], params.shape[0]), axis=0) # int-term
    C = 1/pi # prior-term
    gradPi = A - B

    # mu
    A = (pi[np.newaxis,:]*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis]*\
        Sinv_sf_delta
    A = np.sum(A, axis=0) # i-term (nComponent x 2)
    B = (I_jl)[:,np.newaxis] * \
        Sinv_sum_delta
    B = np.sum(B.reshape(df_params.shape[0], params.shape[0], 2), axis=0) # int-term
    C = -l0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1) # NIW prior
    gradmu = A - B + C

    #sigma
    diff = -0.5*(Sinv_sf[np.newaxis] - Sinv_sf_delta[...,np.newaxis]*Sinv_sf_delta[...,np.newaxis,:])
    A = (pi*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis,np.newaxis] * diff
    A = np.sum(A, axis=0) # i-term (nComponent x 2 x 2)
    diff = -0.5*(Sinv_sum - Sinv_sum_delta[...,np.newaxis]*Sinv_sum_delta[...,np.newaxis,:])
    B = (I_jl)[:,np.newaxis,np.newaxis] * diff
    B = np.sum(B.reshape(df_params.shape[0], params.shape[0], 2, 2), axis=0) # int-term (nComponent x 2 x 2)
    C = -((nu0+4)/2.)*Sinv_sf - (l0/2.)*Sinv_delta_mumu0[...,np.newaxis]*Sinv_delta_mumu0[...,np.newaxis,:]\
        + (1./2.) * np.matmul(Sinv_sf, np.matmul(Psi0[np.newaxis,...], Sinv_sf))
    gradS = A - B + C

    grad = np.zeros((params.shape[0],6))
    grad[:,:2] = gradmu
    grad[:,2] = gradS[:,0,0]
    grad[:,3] = gradS[:,1,1]
    grad[:,4] = 2*gradS[:,0,1]*np.sqrt(params[:,2]*params[:,3])*2*p**2*e_alpha
    grad[:,5] = gradPi * 2*np.pi*np.sqrt(Sdet_sf) * p_pi**2 * e_alpha_pi

    # DF priors
    # N df
    A = 1/N
    C = np.log(float(Nphot)/N)
    gradN = A+C
    # pi df
    A = np.sum(m_ij_df/m_i_df[:,np.newaxis], axis=0) # i-term
    B = np.sum((I_jl/pi_df[df_idx]).reshape(params.shape[0], df_params.shape[0]), axis=0) # int-term
    C = (conc_df/np.sum(conc_df)-1)/pi_df
    gradPi = A - B + C
    # weight
    gradW = gradN/pi_df + gradPi/N
    #print('A', A)
    #print('-B', -B)
    #print('C', C)
    #print(gradN/pi_df,  gradPi/N)
    #print(gradW)

    # mu df
    A = np.sum( (pi_df[np.newaxis,:]*(m_ij_df/m_i_df[:,np.newaxis]))[...,np.newaxis]*\
                Sinv_df_delta, axis=0)
    B = np.sum(((I_jl)[:,np.newaxis] * Sinv_sum_delta).reshape(df_params.shape[0], params.shape[0], 2), axis=1)
    C = -ldf[:,np.newaxis] * np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1) # NIW prior
    gradmu = A - B + C

    #sigma DF
    diff = -0.5*(Sinv_df[np.newaxis] - Sinv_df_delta[...,np.newaxis]*Sinv_df_delta[...,np.newaxis,:])
    A = (pi_df*(m_ij_df/m_i_df[:,np.newaxis]))[...,np.newaxis,np.newaxis] * diff
    A = np.sum(A, axis=0) # i-term (nComponent x 2 x 2)
    diff = -0.5*(Sinv_sum - Sinv_sum_delta[...,np.newaxis]*Sinv_sum_delta[...,np.newaxis,:])
    B = (I_jl)[:,np.newaxis,np.newaxis] * diff
    B = np.sum(B.reshape(df_params.shape[0], params.shape[0], 2, 2), axis=1) # int-term (nComponent x 2 x 2)
    C = -((nudf[:,np.newaxis,np.newaxis]+4)/2.)*Sinv_df - (ldf[:,np.newaxis,np.newaxis]/2.)*(Sinv_delta_mumudf[...,np.newaxis]*Sinv_delta_mumudf[...,np.newaxis,:])\
        + (1./2.) * np.matmul(Sinv_df, np.matmul(Psidf, Sinv_df))
    gradS = A - B + C

    grad_df = np.zeros((df_params.shape[0],6))
    grad_df[:,:2] = gradmu
    grad_df[:,2] = gradS[:,0,0]
    grad_df[:,3] = gradS[:,1,1]
    grad_df[:,4] = gradS[:,0,1]*np.sqrt(df_params[:,2]*df_params[:,3])
    grad_df[:,5] = gradW

    grad = np.hstack((grad.flatten(), grad_df.flatten()))

    return  nlnP, -grad
def calc_nlnP_FullBHM(params, Xsf, NIWprior, Post_df,
                    get_grad=True, stdout=False,test=False, component='',
                    component_order_sf=None, component_order_df=None):

    ndim = len(params)
    # Prior on selection function components
    m0, l0, Psi0, nu0 = NIWprior
    # Prior on distribution function components
    Nphot, conc_df, mdf, ldf, Psidf, nudf = Post_df
    conc_df = conc_df/np.sum(conc_df)
    ncomponents_df = mdf.shape[0]

    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    params[:,2:4] = np.abs(params[:,2:4])
    df_params = params[-ncomponents_df:].copy()
    params = params[:-ncomponents_df].copy()
    N = np.sum(df_params[:,5])
    # Prior on df_params
    if (np.sum(np.abs(df_params[...,4])>=1)>0)|(np.sum(df_params[:,5]<0)>0)|(N<0):
        if get_grad: return 1e10, np.zeros(ndim)
        else: return 1e10
    # Prior on sf params
    if component_order_sf is not None:
        if bool(np.product((np.argsort(params[:,0])==component_order_sf[:,0])))&\
           bool(np.product((np.argsort(params[:,1])==component_order_sf[:,1]))):
            pass
        else:
            #print('SF disordered')
            if get_grad: return 1e10, np.zeros(ndim)
            else: return 1e10
    if component_order_df is not None:
        if bool(np.product((np.argsort(df_params[:,0])==component_order_df[:,0])))&\
           bool(np.product((np.argsort(df_params[:,1])==component_order_df[:,1]))):
            pass
        else:
            #print('DF disordered')
            if get_grad: return 1e10, np.zeros(ndim)
            else: return 1e10
    # means
    df_idx = np.repeat(np.arange(df_params.shape[0]), params.shape[0])
    sf_idx = np.tile(np.arange(params.shape[0]), df_params.shape[0])
    # covariances
    e_alpha = np.exp(-params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr = (2*p - 1)
    cov = np.sqrt(params[...,2]*params[...,3])*corr
    S_sf = np.moveaxis(np.array([[params[...,2], cov], [cov, params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf.copy())
    delta_sf = Xsf[:,np.newaxis,:]-params[:,:2][np.newaxis,:,:]
    Sinv_sf_delta = np.sum(Sinv_sf[np.newaxis,...]*delta_sf[...,np.newaxis], axis=2).copy()
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-params[...,5])
    p_pi = 1./(1+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * p_pi

    # Now for the DF
    cov_df = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], cov_df], [cov_df, df_params[...,3]]]), -1, 0)
    Sinv_df, Sdet_df = quick_invdet(S_df)
    delta_df = Xsf[:,np.newaxis,:]-df_params[:,:2][np.newaxis,:,:]
    Sinv_df_delta = np.sum(Sinv_df[np.newaxis,...]*delta_df[...,np.newaxis], axis=2)
    #weights
    #pi_df = df_params[:,5].copy()#/N
    w_df = df_params[:,5].copy()
    pi_df = w_df.copy()/np.sum(w_df)
    #print(np.log(Sdet_df), nudf)

    # Unscaled parameters
    params_original = params.copy()
    params_original[:,4] = corr.copy()
    params_original[:,5] = pi
    # Max SF prior
    gmm_maxima = gradient_rootfinder(params_original, Sinv_sf, 1/(2*np.pi*np.sqrt(Sdet_sf)))
    #print(gmm_maxima)
    #print(gmm_maxima)
    if gmm_maxima>1:
        if get_grad: return 1e10, np.zeros(ndim)
        else: return 1e10

    # Joint sf-df model
    Sinv_sum, Sdet_sum = quick_invdet(S_sf[sf_idx]+S_df[df_idx])
    Sinv_sum_delta = np.sum(Sinv_sum*(df_params[:,:2][df_idx] - params[:,:2][sf_idx])[...,np.newaxis], axis=1)

    # Calculation for later
    Sinv_dd_Sinv = Sinv_sf_delta[...,np.newaxis]*Sinv_sf_delta[...,np.newaxis,:]
    Sinv_dd_Sinv_df = Sinv_df_delta[...,np.newaxis]*Sinv_df_delta[...,np.newaxis,:]
    delta_mumu0 = params[:,:2]-m0[np.newaxis,:]
    Sinv_delta_mumu0 = np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1)
    delta_mumudf = df_params[:,:2]-mdf
    Sinv_delta_mumudf = np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1)
    SddS_mumudf = Sinv_delta_mumudf[...,np.newaxis]*Sinv_delta_mumudf[...,np.newaxis,:]

    # Likelihood
    # SF star iteration term (Nstar x Ncomponent_sf)
    m_ij = Gaussian_i(delta_sf, Sinv_sf, Sdet_sf)
    m_i = np.sum(pi*m_ij, axis=1)
    # DF star iteration term (Nstar x Ncomponent_sf)
    m_ij_df = Gaussian_i(delta_df, Sinv_df, Sdet_df)
    m_i_df = np.sum(w_df*m_ij_df, axis=1)
    # Integral Term
    delta_mumu = params[:,:2][sf_idx] - df_params[:,:2][df_idx]
    I_jl = Gaussian_int(delta_mumu, Sinv_sum, Sdet_sum)  * pi[sf_idx] * df_params[:,5][df_idx]
    I = np.sum(I_jl)

    # Prior
    Prior0 = (-(nu0+4.)/2.) * np.log(Sdet_sf)
    Priormu = (-l0/2.) * np.sum(delta_mumu0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1), axis=1)
    PriorS = (-1/2.) * np.trace(np.matmul(Psi0[np.newaxis,...], Sinv_sf), axis1=-2, axis2=-1)
    Prior = np.sum(Prior0 + Priormu + PriorS)

    PriorN_df = N*np.log(Nphot/N) - (Nphot-N)
    PriorPi_df = (conc_df - 1) * np.log(pi_df)
    Prior0_df = (-(nudf+4.)/2.) * np.log(Sdet_df)
    Priormu_df = (-ldf/2.) * np.sum(delta_mumudf * \
                                    np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1), axis=1)
    PriorS_df = (-1/2.) * np.trace(np.matmul(Psidf, Sinv_df), axis1=-2, axis2=-1)
    Prior_df = PriorN_df + np.sum(PriorPi_df + Prior0_df + Priormu_df + PriorS_df)

    #print(np.sum(np.log(m_i_df)), np.sum(np.log(m_i)), -np.sum(I), Prior, Prior_df)
    nlnP = - ( np.sum(np.log(m_i_df)) + np.sum(np.log(m_i)) - np.sum(I) + Prior + Prior_df)
    #print( 'Probability: ', np.sum(np.log(m_i_df)), np.sum(np.log(m_i)), - np.sum(I), Prior, Prior_df)
    #print( 'Prior df: ', PriorN_df, np.sum(PriorPi_df), np.sum(Prior0_df), np.sum(Priormu_df), np.sum(PriorS_df))
    if (not get_grad) and (not test):
        return nlnP


    # Gradients
    # Pi
    xgrad_pi = np.sum(m_ij/m_i[:,np.newaxis], axis=0) # i-term
    Igrad_pi = -np.sum((I_jl/pi[sf_idx]).reshape(df_params.shape[0], params.shape[0]), axis=0) # int-term
    prgrad_pi = 0.#1/pi # prior-term

    # mu
    xgrad_mu = (pi[np.newaxis,:]*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis]*\
        Sinv_sf_delta
    xgrad_mu = np.sum(xgrad_mu, axis=0) # i-term (nComponent x 2)
    Igrad_mu = -np.sum(((I_jl)[:,np.newaxis] * Sinv_sum_delta).\
                             reshape(df_params.shape[0], params.shape[0], 2), axis=0)
    prgrad_mu = -l0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1) # NIW prior

    #sigma
    diff_xs = 0.5 * (2*Sinv_dd_Sinv - np.eye(2)[np.newaxis,np.newaxis,:,:]*(Sinv_sf_delta**2)[...,np.newaxis])
    xgrad_s = (pi[np.newaxis,:]*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis,np.newaxis]*diff_xs
    xgrad_s = np.sum(xgrad_s, axis=0) # i-term (nComponent x 2 x 2)
    Sinv_dd_Sinv_sum = Sinv_sum_delta[...,np.newaxis]*Sinv_sum_delta[...,np.newaxis,:]
    diff = 0.5*(Sinv_dd_Sinv_sum + Sinv_sf[sf_idx] - Sinv_sum)
    diff_diag = np.eye(2)[np.newaxis,np.newaxis,...]*np.diagonal(diff, axis1=-2,axis2=-1)[...,np.newaxis]
    diff_Is = 2*diff-diff_diag
    Igrad_s = (I_jl)[:,np.newaxis,np.newaxis] * diff_Is
    Igrad_s = -np.sum(Igrad_s.reshape(df_params.shape[0], params.shape[0], 2, 2), axis=0) # int-term (nComponent x 2 x 2)
    prgrad_s = -((nu0+4)/2.)*Sinv_sf - (l0/2.)*Sinv_delta_mumu0[...,np.newaxis]*Sinv_delta_mumu0[...,np.newaxis,:]\
        + (1./2.) * np.matmul(Sinv_sf, np.matmul(Psi0[np.newaxis,...], Sinv_sf))
    pr_diag = np.eye(2)[np.newaxis,...]*np.diagonal(prgrad_s, axis1=-2,axis2=-1)[...,np.newaxis]
    prgrad_s = 2*prgrad_s - pr_diag

    # DF priors
    # N df
    # Not using this part for x now
    xgrad_Ndf = Xsf.shape[0]/N
    prgrad_Ndf = np.log(float(Nphot)/N)
    gradN = xgrad_Ndf + prgrad_Ndf
    # pi df
    xgrad_pidf = np.sum(m_ij_df/m_i_df[:,np.newaxis], axis=0)# i-term
    Igrad_pidf = -np.sum((I_jl/w_df[df_idx]).reshape(df_params.shape[0],params.shape[0]), axis=1) # int-term
    prgrad_pidf = (conc_df-1)/(pi_df)
    prgrad_pidf = (1/N) * (prgrad_pidf - np.sum(prgrad_pidf*pi_df))

    # mu df
    xgrad_mudf = np.sum( (w_df[np.newaxis,:]*(m_ij_df/m_i_df[:,np.newaxis]))[...,np.newaxis]*\
                Sinv_df_delta, axis=0)
    Igrad_mudf = np.sum(((I_jl)[:,np.newaxis] * Sinv_sum_delta).\
                             reshape(df_params.shape[0], params.shape[0], 2), axis=1)
    prgrad_mudf = -ldf[:,np.newaxis] * np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1) # NIW prior

    #sigma DF
    diff = -0.5*(Sinv_df - Sinv_dd_Sinv_df)
    diff_diag = np.eye(2)[np.newaxis,np.newaxis,...]*np.diagonal(diff, axis1=-2,axis2=-1)[...,np.newaxis]
    diff = 2*diff-diff_diag
    xgrad_sdf = (w_df*(m_ij_df/m_i_df[:,np.newaxis]))[...,np.newaxis,np.newaxis] * diff
    xgrad_sdf = np.sum(xgrad_sdf, axis=0) # i-term (nComponent x 2 x 2)
    diff = 0.5*(Sinv_dd_Sinv_sum - Sinv_sum)
    diff_diag = np.eye(2)[np.newaxis,np.newaxis,...]*np.diagonal(diff, axis1=-2,axis2=-1)[...,np.newaxis]
    diff_Is = 2*diff-diff_diag
    Igrad_sdf = (I_jl)[:,np.newaxis,np.newaxis] * diff_Is
    Igrad_sdf = -np.sum(Igrad_sdf.reshape(df_params.shape[0], params.shape[0], 2, 2), axis=1) # int-term (nComponent x 2 x 2)
    prgrad_sdf = -((nudf[:,np.newaxis,np.newaxis]+4)/2.)*Sinv_df \
                 + (ldf[:,np.newaxis,np.newaxis]/2.)*(SddS_mumudf)\
                 + (1./2.) * np.matmul(Sinv_df, np.matmul(Psidf, Sinv_df))
    pr_sdf_diag = np.eye(2)[np.newaxis,...]*np.diagonal(prgrad_sdf, axis1=-2,axis2=-1)[...,np.newaxis]
    prgrad_sdf = 2*prgrad_sdf - pr_sdf_diag

    if test:

        if component=='xsf':
            nlnP, grad = gradtest_x(params, df_params, m_i, xgrad_mu, xgrad_s, xgrad_pi,
                                p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov)
        if component=='xdf':
            nlnP, grad = gradtest_xdf(params, df_params, m_i_df,
                        xgrad_mudf, xgrad_sdf, xgrad_pidf, xgrad_Ndf,
                        p, e_alpha, p_pi, e_alpha_pi, N, pi_df, Sdet_sf, cov_df)
        if component=='integral':
            nlnP, grad = gradtest_I(params, df_params, I,
                                Igrad_mu, Igrad_s, Igrad_pi,
                                Igrad_mudf, Igrad_sdf, Igrad_pidf,
                                p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov, cov_df)
        if component=='prior_sf':
            nlnP, grad = gradtest_prior(params, df_params, Prior,
                        prgrad_mu, prgrad_s, prgrad_pi,
                        p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov)
        if component=='prior_df':
            nlnP, grad = gradtest_priordf(params, df_params, Prior_df,
                        prgrad_mudf, prgrad_sdf, prgrad_pidf, prgrad_Ndf,
                        p, e_alpha, p_pi, e_alpha_pi, N, pi_df, Sdet_sf, cov_df)
        if component=='full':
            pass
        if not get_grad: return nlnP
        return nlnP, -grad

    gradS = xgrad_s + Igrad_s + prgrad_s
    grad = np.zeros((params.shape[0],6))
    grad[:,:2] = xgrad_mu + Igrad_mu + prgrad_mu
    grad[:,2] = gradS[:,0,0] + gradS[:,0,1]*cov/(2*params[:,2])
    grad[:,3] = gradS[:,1,1] + gradS[:,0,1]*cov/(2*params[:,3])
    grad[:,4] = gradS[:,0,1]*np.sqrt(params[:,2]*params[:,3])* 2*e_alpha/(1+e_alpha)**2
    grad[:,5] = (xgrad_pi + Igrad_pi + prgrad_pi) * 2*np.pi*np.sqrt(Sdet_sf) * p_pi**2 * e_alpha_pi

    gradS_df = xgrad_sdf + Igrad_sdf + prgrad_sdf
    grad_df = np.zeros((df_params.shape[0],6))
    grad_df[:,:2] = xgrad_mudf + Igrad_mudf + prgrad_mudf
    grad_df[:,2] = gradS_df[:,0,0] + gradS_df[:,0,1]*cov_df/(2*df_params[:,2])
    grad_df[:,3] = gradS_df[:,1,1] + gradS_df[:,0,1]*cov_df/(2*df_params[:,3])
    grad_df[:,4] = gradS_df[:,0,1]*np.sqrt(df_params[:,2]*df_params[:,3])
    grad_df[:,5] = xgrad_pidf + Igrad_pidf + prgrad_pidf + prgrad_Ndf

    grad = np.hstack((grad.flatten(), grad_df.flatten()))

    return  nlnP, -grad
def calc_nlnP_SFOnly(params, Xsf, NIWprior, df_params,
                    get_grad=True, stdout=False,test=False, L_only=False,
                    component='',
                    component_order=None):

    ndim = len(params)
    # Prior on selection function components
    m0, l0, Psi0, nu0 = NIWprior

    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    params[:,2:4] = np.abs(params[:,2:4])
    N = np.sum(df_params[:,5])
    # Prior on df_params
    if (np.sum(np.abs(df_params[...,4])>=1)>0)|(np.sum(df_params[:,5]<0)>0)|(N<0):
        if get_grad: return 1e10, np.zeros(ndim)
        else: return 1e10
    # Prior on sf params
    if component_order is not None:
        if bool(np.product((np.argsort(params[:,0])==component_order[:,0])))&\
           bool(np.product((np.argsort(params[:,1])==component_order[:,1]))):
            pass
        else:
            if get_grad: return 1e10, np.zeros(ndim)
            else: return 1e10


    # means
    df_idx = np.repeat(np.arange(df_params.shape[0]), params.shape[0])
    sf_idx = np.tile(np.arange(params.shape[0]), df_params.shape[0])
    # covariances
    e_alpha = np.exp(-params[...,4])
    p = (1-1e-10)/(1+e_alpha)
    corr = (2*p - 1)
    cov = np.sqrt(params[...,2]*params[...,3])*corr
    S_sf = np.moveaxis(np.array([[params[...,2], cov], [cov, params[...,3]]]), -1, 0)
    Sinv_sf, Sdet_sf = quick_invdet(S_sf.copy())
    delta_sf = Xsf[:,np.newaxis,:]-params[:,:2][np.newaxis,:,:]
    Sinv_sf_delta = np.sum(Sinv_sf[np.newaxis,...]*delta_sf[...,np.newaxis], axis=2).copy()
    #weights
    # Logit correction of params[:,5] - [-inf, inf] --> [0, rt(det(2.pi.S))]
    e_alpha_pi = np.exp(-params[...,5])
    p_pi = 1./(1+e_alpha_pi)
    pi = 2*np.pi*np.sqrt(Sdet_sf) * p_pi

    # Now for the DF
    cov_df = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], cov_df], [cov_df, df_params[...,3]]]), -1, 0)
    Sinv_df, Sdet_df = quick_invdet(S_df)
    delta_df = Xsf[:,np.newaxis,:]-df_params[:,:2][np.newaxis,:,:]
    Sinv_df_delta = np.sum(Sinv_df[np.newaxis,...]*delta_df[...,np.newaxis], axis=2)
    #weights
    #pi_df = df_params[:,5].copy()#/N
    w_df = df_params[:,5].copy()
    pi_df = w_df.copy()/np.sum(w_df)

    # Unscaled parameters
    params_original = params.copy()
    params_original[:,4] = corr.copy()
    params_original[:,5] = pi
    # Max SF prior
    gmm_maxima = gradient_rootfinder(params_original, Sinv_sf, 1/(2*np.pi*np.sqrt(Sdet_sf)))
    #print(gmm_maxima)
    if gmm_maxima>1:
        if get_grad: return 1e10, np.zeros(ndim)
        else: return 1e10

    # Joint sf-df model
    Sinv_sum, Sdet_sum = quick_invdet(S_sf[sf_idx]+S_df[df_idx])
    Sinv_sum_delta = np.sum(Sinv_sum*(df_params[:,:2][df_idx] - params[:,:2][sf_idx])[...,np.newaxis], axis=1)

    # Calculation for later
    Sinv_dd_Sinv = Sinv_sf_delta[...,np.newaxis]*Sinv_sf_delta[...,np.newaxis,:]
    delta_mumu0 = params[:,:2]-m0[np.newaxis,:]
    Sinv_delta_mumu0 = np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1)

    # Likelihood
    # SF star iteration term (Nstar x Ncomponent_sf)
    m_ij = Gaussian_i(delta_sf, Sinv_sf, Sdet_sf)
    m_i = np.sum(pi*m_ij, axis=1)
    # DF star iteration term (Nstar x Ncomponent_sf)
    m_ij_df = Gaussian_i(delta_df, Sinv_df, Sdet_df)
    m_i_df = np.sum(w_df*m_ij_df, axis=1)
    # Integral Term
    delta_mumu = params[:,:2][sf_idx] - df_params[:,:2][df_idx]
    I_jl = Gaussian_int(delta_mumu, Sinv_sum, Sdet_sum)  * pi[sf_idx] * df_params[:,5][df_idx]
    I = np.sum(I_jl)

    # Prior
    Prior0 = (-(nu0+4.)/2.) * np.log(Sdet_sf)
    Priormu = (-l0/2.) * np.sum(delta_mumu0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1), axis=1)
    PriorS = (-1/2.) * np.trace(np.matmul(Psi0[np.newaxis,...], Sinv_sf), axis1=-2, axis2=-1)
    Prior = np.sum(Prior0 + Priormu + PriorS)

    nlnP = - ( np.sum(np.log(m_i)) - np.sum(I) + Prior )
    if L_only:
        return - (np.sum(np.log(m_i)) - np.sum(I))
    if (not get_grad) and (not test):
        return nlnP

    # Gradients
    # Pi
    xgrad_pi = np.sum(m_ij/m_i[:,np.newaxis], axis=0) # i-term
    Igrad_pi = -np.sum((I_jl/pi[sf_idx]).reshape(df_params.shape[0], params.shape[0]), axis=0) # int-term
    prgrad_pi = 0.#1/pi # prior-term

    # mu
    xgrad_mu = (pi[np.newaxis,:]*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis]*\
        Sinv_sf_delta
    xgrad_mu = np.sum(xgrad_mu, axis=0) # i-term (nComponent x 2)
    Igrad_mu = -np.sum(((I_jl)[:,np.newaxis] * Sinv_sum_delta).\
                             reshape(df_params.shape[0], params.shape[0], 2), axis=0)
    prgrad_mu = -l0 * np.sum(Sinv_sf * delta_mumu0[...,np.newaxis], axis=1) # NIW prior

    #sigma
    diff_xs = 0.5 * (2*Sinv_dd_Sinv - np.eye(2)[np.newaxis,np.newaxis,:,:]*(Sinv_sf_delta**2)[...,np.newaxis])
    xgrad_s = (pi[np.newaxis,:]*(m_ij/m_i[:,np.newaxis]))[...,np.newaxis,np.newaxis]*diff_xs
    xgrad_s = np.sum(xgrad_s, axis=0) # i-term (nComponent x 2 x 2)
    Sinv_dd_Sinv_sum = Sinv_sum_delta[...,np.newaxis]*Sinv_sum_delta[...,np.newaxis,:]
    diff = 0.5*(Sinv_dd_Sinv_sum + Sinv_sf[sf_idx] - Sinv_sum)
    diff_diag = np.eye(2)[np.newaxis,np.newaxis,...]*np.diagonal(diff, axis1=-2,axis2=-1)[...,np.newaxis]
    diff_Is = 2*diff-diff_diag
    Igrad_s = (I_jl)[:,np.newaxis,np.newaxis] * diff_Is
    Igrad_s = -np.sum(Igrad_s.reshape(df_params.shape[0], params.shape[0], 2, 2), axis=0) # int-term (nComponent x 2 x 2)
    prgrad_s = -((nu0+4)/2.)*Sinv_sf - (l0/2.)*Sinv_delta_mumu0[...,np.newaxis]*Sinv_delta_mumu0[...,np.newaxis,:]\
        + (1./2.) * np.matmul(Sinv_sf, np.matmul(Psi0[np.newaxis,...], Sinv_sf))
    pr_diag = np.eye(2)[np.newaxis,...]*np.diagonal(prgrad_s, axis1=-2,axis2=-1)[...,np.newaxis]
    prgrad_s = 2*prgrad_s - pr_diag

    if test:
        if component=='xsf':
            nlnP, grad = gradtest_x(params, df_params, m_i, xgrad_mu, xgrad_s, xgrad_pi,
                                p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov)
        if component=='integral':
            nlnP, grad = gradtest_I(params, df_params, I,
                                Igrad_mu, Igrad_s, Igrad_pi,
                                Igrad_mudf, Igrad_sdf, Igrad_pidf,
                                p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov, cov_df)
        if component=='prior_sf':
            nlnP, grad = gradtest_prior(params, df_params, Prior,
                        prgrad_mu, prgrad_s, prgrad_pi,
                        p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov)
        if component=='full':
            pass
        if not get_grad: return nlnP
        return nlnP, -grad

    gradS = xgrad_s + Igrad_s + prgrad_s
    grad = np.zeros((params.shape[0],6))
    grad[:,:2] = xgrad_mu + Igrad_mu + prgrad_mu
    grad[:,2] = gradS[:,0,0] + gradS[:,0,1]*cov/(2*params[:,2])
    grad[:,3] = gradS[:,1,1] + gradS[:,0,1]*cov/(2*params[:,3])
    grad[:,4] = gradS[:,0,1]*np.sqrt(params[:,2]*params[:,3])* 2*e_alpha/(1+e_alpha)**2
    grad[:,5] = (xgrad_pi + Igrad_pi + prgrad_pi) * 2*np.pi*np.sqrt(Sdet_sf) * p_pi**2 * e_alpha_pi
    grad = grad.flatten()

    return  nlnP, -grad
def calc_nlnP_priorDF(df_params, Post_df,
                    get_grad=True, stdout=False,test=False, component='',
                    component_order_sf=None, component_order_df=None):

    ndim = len(df_params)
    # Prior on distribution function components
    Nphot, conc_df, mdf, ldf, Psidf, nudf = Post_df
    conc_df = conc_df/np.sum(conc_df)
    ncomponents_df = mdf.shape[0]

    # Parameters - transform to means, covariances and weights
    df_params = np.reshape(df_params, (-1,6))
    N = np.sum(df_params[:,5])
    # Prior on df_params
    if (np.sum(np.abs(df_params[...,4])>=1)>0)|(np.sum(df_params[:,5]<0)>0)|(N<0):
        if get_grad: return 1e10, np.zeros(ndim)
        else: return 1e10
    if component_order_df is not None:
        if bool(np.product((np.argsort(df_params[:,0])==component_order_df[:,0])))&\
           bool(np.product((np.argsort(df_params[:,1])==component_order_df[:,1]))):
            pass
        else:
            print('DF disordered')
            if get_grad: return 1e10, np.zeros(ndim)
            else: return 1e10

    # Now for the DF
    cov_df = np.sqrt(df_params[...,2]*df_params[...,3])*df_params[...,4]
    S_df = np.moveaxis(np.array([[df_params[...,2], cov_df], [cov_df, df_params[...,3]]]), -1, 0)
    Sinv_df, Sdet_df = quick_invdet(S_df)
    #weights
    #pi_df = df_params[:,5].copy()#/N
    w_df = df_params[:,5].copy()
    pi_df = w_df.copy()/np.sum(w_df)
    print('lndet: ',np.log(Sdet_df))
    print('nudf: ',nudf)

    # Calculation for later
    delta_mumudf = df_params[:,:2]-mdf
    Sinv_delta_mumudf = np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1)
    SddS_mumudf = Sinv_delta_mumudf[...,np.newaxis]*Sinv_delta_mumudf[...,np.newaxis,:]


    PriorN_df = N*np.log(Nphot/N) - (Nphot-N)
    PriorPi_df = (conc_df - 1) * np.log(pi_df)
    Prior0_df = (-(nudf+4.)/2.) * np.log(Sdet_df)
    Priormu_df = (-ldf/2.) * np.sum(delta_mumudf * \
                                    np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1), axis=1)
    PriorS_df = (-1/2.) * np.trace(np.matmul(Psidf, Sinv_df), axis1=-2, axis2=-1)
    Prior_df = PriorN_df + np.sum(PriorPi_df + Prior0_df + Priormu_df + PriorS_df)

    #print(np.sum(np.log(m_i_df)), np.sum(np.log(m_i)), -np.sum(I), Prior, Prior_df)
    nlnP = - ( Prior_df)
    print( 'Prior df: ', PriorN_df, np.sum(PriorPi_df), np.sum(Prior0_df), np.sum(Priormu_df), np.sum(PriorS_df))
    if (not get_grad) and (not test):
        return nlnP

    # Gradients
    prgrad_Ndf = np.log(float(Nphot)/N)
    gradN = xgrad_Ndf + prgrad_Ndf
    # pi df
    prgrad_pidf = (conc_df-1)/(pi_df)
    prgrad_pidf = (1/N) * (prgrad_pidf - np.sum(prgrad_pidf*pi_df))
    # mu df
    prgrad_mudf = -ldf[:,np.newaxis] * np.sum(Sinv_df * delta_mumudf[...,np.newaxis], axis=1) # NIW prior
    #sigma DF
    prgrad_sdf = -((nudf[:,np.newaxis,np.newaxis]+4)/2.)*Sinv_df \
                 + (ldf[:,np.newaxis,np.newaxis]/2.)*(SddS_mumudf)\
                 + (1./2.) * np.matmul(Sinv_df, np.matmul(Psidf, Sinv_df))
    pr_sdf_diag = np.eye(2)[np.newaxis,...]*np.diagonal(prgrad_sdf, axis1=-2,axis2=-1)[...,np.newaxis]
    prgrad_sdf = 2*prgrad_sdf - pr_sdf_diag

    gradS_df = prgrad_sdf
    grad_df = np.zeros((df_params.shape[0],6))
    grad_df[:,:2] = prgrad_mudf
    grad_df[:,2] = gradS_df[:,0,0] + gradS_df[:,0,1]*cov_df/(2*df_params[:,2])
    grad_df[:,3] = gradS_df[:,1,1] + gradS_df[:,0,1]*cov_df/(2*df_params[:,3])
    grad_df[:,4] = gradS_df[:,0,1]*np.sqrt(df_params[:,2]*df_params[:,3])
    grad_df[:,5] = prgrad_pidf + prgrad_Ndf

    return  nlnP, -grad_df.flatten()
def lnlike(Xdf, params):

    function = lambda a, b: bivGaussMixture(params, a, b)
    model = bivGaussMixture(params, Xdf[:,0], Xdf[:,1])

    contPoints = np.sum( np.log(model) )

    # Integral of the smooth function over the entire region
    contInteg = np.sum(params[:,5])
    #print bivGauss_analytical_approx(params, self.rngx_s, self.rngy_s), self.rngx_s, self.rngy_s

    lnL = contPoints - contInteg

    return lnL
def BIC(n, k, lnL):
    return k*np.log(n) - 2*lnL
def AIC(n, k, lnL):
    return 2*k - 2*lnL
def AICc(n, k, lnL):
    return 2*k - 2*lnL + (2*k**2 + 2*k)/(n-k-1)

# Optimization methods
def TNC_sf(Xsf, priorParams, df_params, max_components=10, stdout=False,
            init='BGM', Xdf=None):

    bic_vals = np.zeros(max_components) + np.inf
    post_vals = np.zeros(max_components) - np.inf
    sf_params_n = {}
    for i in range(2, max_components):

        n_component=i

        # Simple GMM
        if not init=='best':
            if init=='BGM':
                gmm = mixture.BayesianGaussianMixture(n_component, n_init=1,
                                                      init_params='kmeans', tol=1e-5, max_iter=1000)
                gmm.fit(Xsf)
                params_i = get_sfparams_logit(gmm, n_component)
            elif init=='kmeans':
                weights = 1/bivGaussMixture(df_params, Xsf[:,0], Xsf[:,1])
                # K-Means clustering of initialisation
                #print(Xsf.shape, Xdf.shape)
                params=kmeans_init(Xsf, Xdf, i, weights=weights)
                #print(params)
                params_i=transform_sfparams_invlogit(params)

            opt = op.minimize(calc_nlnP_grad_pilogit_NIW,  params_i,
                                          args=(Xsf, priorParams, df_params), method='TNC',
                                          jac=True, options={'maxiter':1000}, tol=1e-5)
            nlnp = calc_nlnP_grad_pilogit_NIW(opt.x, Xsf, priorParams, df_params)[0]

        else:
            params_pilogit = []
            # BGM
            gmm = mixture.BayesianGaussianMixture(n_component, n_init=1,
                                                  init_params='kmeans', tol=1e-5, max_iter=1000)
            gmm.fit(Xsf)
            params_bgm_pilogit = get_sfparams_logit(gmm, n_component)
            params_pilogit.insert(0, params_bgm_pilogit)

            # KMS
            weights = 1/bivGaussMixture(df_params, Xsf[:,0], Xsf[:,1])
            # K-Means clustering of initialisation
            #print(Xsf.shape, Xdf.shape)
            params=kmeans_init(Xsf, Xdf, i, weights=weights)
            #print(params)
            params_kms_pilogit=transform_sfparams_invlogit(params)
            params_pilogit.insert(0, params_kms_pilogit)

            nlnp_trials = np.zeros(2)
            opt_trials = [None, None]
            methods = ['KMS', 'BGM']
            for ii in range(2):
                opt = op.minimize(calc_nlnP_grad_pilogit_NIW,  params_pilogit[ii],
                                              args=(Xsf, priorParams, df_params), method='TNC',
                                              jac=True, options={'maxiter':1000}, tol=1e-5)
                nlnp_trials[ii] = calc_nlnP_grad_pilogit_NIW(opt.x, Xsf, priorParams, df_params)[0]
                opt_trials[ii] = opt

            nlnp = np.min(nlnp_trials)
            method = methods[np.argmin(nlnp_trials)]
            opt = opt_trials[np.argmin(nlnp_trials)]
            print(method, nlnp_trials, '  Test: ', calc_nlnP_grad_pilogit_NIW(opt.x, Xsf, priorParams, df_params)[0])

        bic_val = BIC(Xsf.shape[0], i*6, -nlnp)
        bic_vals[i] = bic_val
        post_vals[i] = -nlnp
        if stdout:
            print(opt.success, opt.message)
            print(i, "   BIC: ", bic_val, "   lnP: ", -nlnp)

        sf_params_n[i] = opt.x.reshape(-1,6)

        #if i>1:
        #    if (bic_vals[i]>bic_vals[i-1]) and (bic_vals[i-1]>bic_vals[i-2]):
        #        break

    if stdout:
        print('Best components (posterior): ', np.argmax(post_vals))
        print('Best components (BIC): ', np.argmin(bic_vals))

    return sf_params_n[np.argmin(bic_vals)]
    #return sf_params_n[np.argmin(bic_vals)]
def TNC_sf_HBM(Xsf, priorParams, df_params, df_posterior, max_components=10, stdout=False,
            init='BGM', Xdf=None):

    bic_vals = np.zeros(max_components) + np.inf
    post_vals = np.zeros(max_components) - np.inf
    sf_params_n = {}
    df_newparams_n = {}
    for i in range(2, max_components):

        n_component=i

        # Simple GMM
        if not init=='best':
            if init=='BGM':
                gmm = mixture.BayesianGaussianMixture(n_component, n_init=1,
                                                      init_params='kmeans', tol=1e-5, max_iter=1000)
                gmm.fit(Xsf)
                params_bgm_pilogit = get_sfparams_logit(gmm, n_component)
                sf_params_i = params_bgm_pilogit
            elif init=='kmeans':
                weights = 1/bivGaussMixture(df_params, Xsf[:,0], Xsf[:,1])
                # K-Means clustering of initialisation
                #print(Xsf.shape, Xdf.shape)
                params=kmeans_init(Xsf, Xdf, i, weights=weights)
                #print(params)
                params_kms_pilogit=transform_sfparams_invlogit(params)

                sf_params_i = params_kms_pilogit

            params_i = np.vstack((sf_params_i, df_params))
            opt = op.minimize(calc_nlnP_FullBHM,  params_i,
                                          args=(Xsf, priorParams, df_posterior),
                                          method='TNC',jac=True, options={'maxiter':1000}, tol=1e-5)
            nlnp = calc_nlnP_FullBHM(opt.x, Xsf, priorParams, df_posterior, get_grad=False)

        else:
            params_pilogit = []
            # BGM
            gmm = mixture.BayesianGaussianMixture(n_component, n_init=1,
                                                  init_params='kmeans', tol=1e-5, max_iter=1000)
            gmm.fit(Xsf)
            params_bgm_pilogit = get_sfparams_logit(gmm, n_component)
            params_bgm_i = np.vstack((params_bgm_pilogit, df_params))
            params_pilogit.insert(0, params_bgm_i)

            # KMS
            weights = 1/bivGaussMixture(df_params, Xsf[:,0], Xsf[:,1])
            # K-Means clustering of initialisation
            #print(Xsf.shape, Xdf.shape)
            params=kmeans_init(Xsf, Xdf, i, weights=weights)
            #print(params)
            params_kms_pilogit=transform_sfparams_invlogit(params)
            params_kms_i = np.vstack((params_kms_pilogit, df_params))
            params_pilogit.insert(0, params_kms_i)

            nlnp_trials = np.zeros(2)
            opt_trials = [None, None]
            methods = ['KMS', 'BGM']
            for ii in range(2):
                opt = op.minimize(calc_nlnP_FullBHM,  params_pilogit[ii],
                                              args=(Xsf, priorParams, df_posterior), method='TNC',
                                              jac=True, options={'maxiter':1000}, tol=1e-5)
                nlnp_trials[ii] = calc_nlnP_FullBHM(opt.x, Xsf, priorParams, df_posterior, get_grad=False)
                opt_trials[ii] = opt

            nlnp = np.min(nlnp_trials)
            method = methods[np.argmin(nlnp_trials)]
            opt = opt_trials[np.argmin(nlnp_trials)]
            print(method, nlnp_trials, '  Test: ', calc_nlnP_FullBHM(opt.x, Xsf, priorParams, df_posterior, get_grad=False))

        bic_val = BIC(Xsf.shape[0], i*6, -nlnp)
        bic_vals[i] = bic_val
        post_vals[i] = -nlnp
        if stdout:
            print(opt.success, opt.message)
            print(i, "   BIC: ", bic_val, "   lnP: ", -nlnp)

        sf_params_n[i] = opt.x.reshape(-1,6)[:-df_params.shape[0]]
        df_newparams_n[i] = opt.x.reshape(-1,6)[-df_params.shape[0]:]

    if stdout:
        print('Best components (posterior): ', np.argmax(post_vals))
        print('Best components (BIC): ', np.argmin(bic_vals))

    return sf_params_n[np.argmin(bic_vals)], df_newparams_n[np.argmin(bic_vals)]
def TNC_sf_SFonly(Xsf, priorParams, df_params, max_components=10, stdout=False,
            init='BGM', Xdf=None):

    bic_vals = np.zeros(max_components) + np.inf
    post_vals = np.zeros(max_components) - np.inf
    sf_params_n = {}
    df_newparams_n = {}
    for i in range(1, max_components):

        n_component=i

        # Simple GMM
        if not init=='best':
            if init=='BGM':
                gmm = mixture.BayesianGaussianMixture(n_component, n_init=1,
                                                      init_params='kmeans', tol=1e-5, max_iter=1000)
                gmm.fit(Xsf)
                params_i = get_sfparams_logit(gmm, n_component)
            elif init=='kmeans':
                weights = 1/bivGaussMixture(df_params, Xsf[:,0], Xsf[:,1])
                # K-Means clustering of initialisation
                #print(Xsf.shape, Xdf.shape)
                params=kmeans_init(Xsf, Xdf, i, weights=weights)
                #print(params)
                params_i=transform_sfparams_invlogit(params)

            opt = op.minimize(calc_nlnP_SFOnly,  params_i,
                                          args=(Xsf, priorParams, df_params),
                                          method='TNC',jac=True, options={'maxiter':1000}, tol=1e-5)
            nlnp = calc_nlnP_SFOnly(opt.x, Xsf, priorParams, df_params, get_grad=False)

        else:
            params_pilogit = []
            # BGM
            gmm = mixture.BayesianGaussianMixture(n_component, n_init=1,
                                                  init_params='kmeans', tol=1e-5, max_iter=1000)
            gmm.fit(Xsf)
            params_bgm_i = get_sfparams_logit(gmm, n_component)
            params_pilogit.insert(0, params_bgm_i)

            # KMS
            weights = 1/bivGaussMixture(df_params, Xsf[:,0], Xsf[:,1])
            # K-Means clustering of initialisation
            #print(Xsf.shape, Xdf.shape)
            params=kmeans_init(Xsf, Xdf, i, weights=weights)
            #print(params)
            params_kms_i=transform_sfparams_invlogit(params)
            params_pilogit.insert(0, params_kms_i)

            nlnp_trials = np.zeros(2)
            opt_trials = [None, None]
            methods = ['KMS', 'BGM']
            for ii in range(2):
                opt = op.minimize(calc_nlnP_SFOnly,  params_pilogit[ii],
                                              args=(Xsf, priorParams, df_params),
                                              method='TNC', jac=True, options={'maxiter':1000}, tol=1e-5)
                nlnp_trials[ii] = calc_nlnP_SFOnly(opt.x, Xsf, priorParams, df_params, get_grad=False)
                opt_trials[ii] = opt

            nlnp = np.min(nlnp_trials)
            method = methods[np.argmin(nlnp_trials)]
            opt = opt_trials[np.argmin(nlnp_trials)]
            print(method, nlnp_trials, '  Test: ', calc_nlnP_SFOnly(opt.x, Xsf, priorParams, df_params, get_grad=False))

        nlnL = calc_nlnP_SFOnly(opt.x, Xsf, priorParams, df_params, get_grad=False, L_only=True)
        bic_val = BIC(Xsf.shape[0], i*6, -nlnL)
        bic_vals[i] = bic_val
        post_vals[i] = -nlnp
        if stdout:
            #print(opt.success, opt.message)
            print(i, "   BIC: ", bic_val, "   lnP: ", -nlnp, "   lnL: ", -nlnL)
            print(i, "   BIC: ", bic_val, "   AIC: ", AIC(Xsf.shape[0], i*6, -nlnL), "   AICc: ", AICc(Xsf.shape[0], i*6, -nlnL))

        sf_params_n[i] = opt.x.reshape(-1,6)
    print("Using TNC_sf_SFonly")
    if stdout:
        print('Best components (posterior): ', np.argmax(post_vals))
        print('Best components (BIC): ', np.argmin(bic_vals))

    return sf_params_n[np.argmin(bic_vals)]
def BGMM_df(Xdf, priorParams, max_components=25, stdout=False):

    bic_vals = np.zeros(max_components+1) + np.inf
    df_params_n = {}
    gmm_n = {}

    for i in range(2, max_components+1):

        # Simple GMM
        gmm = mixture.BayesianGaussianMixture(n_components=i, n_init=3,
                                              init_params='kmeans', tol=1e-5, max_iter=2000,
                                              weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=1./float(i),
                                              mean_precision_prior=priorParams[1], degrees_of_freedom_prior=priorParams[3],
                                              covariance_prior= priorParams[2], mean_prior=priorParams[0])
        gmm.fit(Xdf)

        params = get_params(gmm, Xdf.shape[0], i)
        bic_val = BIC(Xdf.shape[0], i*6, lnlike(Xdf, params))
        bic_vals[i] = bic_val
        if stdout:
            print(i, "  BIC: ", bic_val, "   lnP: ", lnlike(Xdf, params))

        df_params_n[i] = params
        gmm_n[i] = gmm

        if i>3:
            if np.product(np.argsort(bic_vals[i-2:i+1]) == np.arange(3)):
                break
            #if (bic_vals[i]>bic_vals[i-1]) and (bic_vals[i-1]>bic_vals[i-2]):
            #    break
    print("Using BGMM_df.")
    if stdout:
        print('Best components: ', np.argmin(bic_vals))

    return df_params_n[np.argmin(bic_vals)], gmm_n[np.argmin(bic_vals)]
def BGMM_emcee_ball(params, Xsf, priorParams, df_params, niter=200):
    print('emcee with %d iterations...' % niter)

    ndim=len(params.flatten())
    nwalkers=ndim*2

    p0 = np.repeat([params,], nwalkers, axis=0)
    p0 = np.random.normal(loc=p0, scale=np.abs(p0/500))
    p0[:,:,2:4] = np.abs(p0[:,:,2:4])

    p0 = p0.reshape(nwalkers, -1)
    foo = lambda a, b, c, d: -calc_nlnP_pilogit_NIW(a, b, c, d)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, foo,
                                    args=(Xsf, priorParams, df_params))
    # Run emcee
    for _ in tqdm(sampler.sample(p0, iterations=niter), total=niter):
        pass

    return sampler
def BGMM_emcee_ball_BHM(params, Xsf, priorParams, DFpost, niter=200):

    ndim=len(params.flatten())
    nwalkers=ndim*2
    print('emcee with %d iterations, %d walkers...' % (niter, nwalkers))

    ncomponents_df = len(DFpost[1])
    # Parameters - transform to means, covariances and weights
    params = np.reshape(params, (-1,6))
    df_params = params[-ncomponents_df:].copy()
    sf_params = params[:-ncomponents_df].copy()
    component_order_sf = np.argsort(sf_params[:,:2], axis=0)
    component_order_df = np.argsort(df_params[:,:2], axis=0)

    p0 = np.repeat([params,], nwalkers, axis=0)
    p0 = np.random.normal(loc=p0, scale=np.abs(p0/500))
    #p0[:,:,2:4] = np.abs(p0[:,:,2:4])

    p0 = p0.reshape(nwalkers, -1)
    foo = lambda a, b, c, d: -calc_nlnP_FullBHM(a, b, c, d, get_grad=False,
                                                component_order_sf=component_order_sf,
                                                component_order_df=component_order_df)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, foo,
                                    args=(Xsf, priorParams, DFpost))
    # Run emcee
    for _ in tqdm(sampler.sample(p0, iterations=niter), total=niter):
        pass

    return sampler
def BGMM_emcee_ball_SFonly(params, Xsf, priorParams, df_params, niter=200):

    ndim=len(params.flatten())
    nwalkers=ndim*2
    print('emcee with %d iterations, %d walkers...' % (niter, nwalkers))

    component_order = np.argsort(params[:,:2], axis=0)

    p0 = np.repeat([params,], nwalkers, axis=0)
    p0 = np.random.normal(loc=p0, scale=np.abs(p0/500))
    #p0[:,:,2:4] = np.abs(p0[:,:,2:4])

    p0 = p0.reshape(nwalkers, -1)
    foo = lambda a, b, c, d: -calc_nlnP_SFOnly(a, b, c, d, get_grad=False, component_order=component_order)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, foo,
                                    args=(Xsf, priorParams, df_params))
    # Run emcee
    for _ in tqdm(sampler.sample(p0, iterations=niter), total=niter):
        pass

    return sampler

def kmeans_init(sample, df_sample, n_components, n_iter=10, max_iter=100, weights=None):

    weighted_kde = not weights is None
    if weights is None: weights=np.ones(len(sample))

    params = np.zeros((n_components, 6))

    kmc = KMeans(n_components, n_init=n_iter, max_iter=max_iter)
    kmc.fit(sample, sample_weight=weights)

    labels = kmc.predict(sample)
    df_labels = kmc.predict(df_sample)

    for i in range(n_components):
        c_sample = sample[labels==i]
        c_df_sample = df_sample[df_labels==i]
        c_weights = weights[labels==i]

        mean = np.sum(c_sample*c_weights[:,np.newaxis], axis=0)/np.sum(c_weights)
        delta = c_sample-mean[np.newaxis, :]
        sigma = np.sum(delta[:,:,np.newaxis]*delta[:,np.newaxis,:]*c_weights[:,np.newaxis,np.newaxis], axis=0)/np.sum(c_weights)

        w = float(len(c_sample))/float(len(c_df_sample))

        if np.sum(labels==i)<=2:
            mu = np.sum(sample*weights[:,np.newaxis], axis=0)/np.sum(weights)
            delta = sample-mu[np.newaxis, :]
            sigma = np.sum(delta[:,:,np.newaxis]*delta[:,np.newaxis,:]*weights[:,np.newaxis,np.newaxis], axis=0)/np.sum(weights)

        multiplier=1.
        params[i, :2] = mean
        params[i, 2:4] = sigma[[0,1],[0,1]]*multiplier
        params[i, 4] = sigma[0,1]/np.sqrt(sigma[0,0]*sigma[1,1])
        params[i, 5] = w * 2*np.pi*np.sqrt(sigma[0,0]*sigma[1,1]*(1-params[i,4]**2)) * multiplier


    sigma = np.array([[params[:,2], np.sqrt(params[:,2]*params[:,3])*params[:,4]],
                      [np.sqrt(params[:,2]*params[:,3])*params[:,4], params[:,3]]])
    sigma = np.moveaxis(sigma, -1, 0)
    sigma_inv, sigma_det = quick_invdet(sigma)
    norm = 1/(2*np.pi*np.sqrt(sigma_det))
    maxima = gradient_rootfinder(params, sigma_inv, norm)
    if maxima>=1:
        params[:,5] *= 0.9/maxima

    return params
def bivGaussMixture(params, x, y):

    '''
    bivGaussMixture - Calculation of bivariate Gaussian distribution.

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
    cov = params[:,4]*np.sqrt(params[:,2]*params[:,3])
    sigma = np.array([[params[:,2], cov],[cov, params[:,3]]])
    sigma = np.moveaxis(sigma, -1, 0)
    weight= params[:,5]

    # Inverse covariance
    #inv_cov = np.linalg.inv(sigma)
    inv_cov, det_cov = quick_invdet(sigma)
    # Separation of X from mean
    X = np.moveaxis(np.repeat([X,], mu.shape[-2], axis=0), 0, -2) - mu

    # X^T * Sigma
    X_ext = X[...,np.newaxis]
    inv_cov = inv_cov[np.newaxis,...]
    X_cov = X_ext*inv_cov
    X_cov = X_cov[...,0,:]+X_cov[...,1,:]
    # X * Sigma * X
    X_cov_X = X_cov*X
    X_cov_X = X_cov_X[:,:,0]+X_cov_X[:,:,1]
    # Exponential
    e = np.exp(-X_cov_X/2)

    # Normalisation term
    #det_cov = np.linalg.det(sigma)
    norm = 1/np.sqrt( ((2*np.pi)**2) * det_cov)

    p = np.sum(weight*norm*e, axis=-1)

    #if np.sum(np.isnan(p))>0: print(params)

    return p.reshape(shape)

# Test likelihood function
def gradtest_all(params_sf, params_df, Xsf, Prior_sf, Prior_df):

    p0 = np.hstack((params_sf.flatten(), params_df.flatten()))

    for component in ['xsf', 'xdf', 'integral', 'prior_sf', 'prior_df', 'full']:
        print(component)
        grad_anc = calc_nlnP_FullBHM(p0, Xsf, Prior_sf, Prior_df, component=component)[1].reshape(-1,6)
        like = lambda params: calc_nlnP_FullBHM(params, Xsf, Prior_sf, Prior_df, component=component)[0]
        grad_app = op.approx_fprime(p0, like, 0.001).reshape(-1, 6)
        for i in range(grad_anc.shape[0]):
            print(grad_anc[i])
            print(grad_app[i])
            print("")
        print("")
def gradtest_like_all(params_sf, params_df, Xsf, Prior_sf, Prior_df):

    p0 = np.hstack((params_sf.flatten(), params_df.flatten()))

    for component in ['xsf', 'xdf', 'integral', 'prior_sf', 'prior_df', 'full']:
        print(component)
        grad_anc = calc_nlnP_FullBHM(p0, Xsf, Prior_sf, Prior_df, component=component)[1].reshape(-1,6)
        like = lambda params: calc_nlnP_FullBHM(params, Xsf, Prior_sf, Prior_df, component=component, get_grad=False)
        grad_app = op.approx_fprime(p0, like, 0.001).reshape(-1, 6)
        for i in range(grad_anc.shape[0]):
            print(grad_anc[i])
            print(grad_app[i])
            print("")
        print("")
def gradtest_like_all_sfonly(params_sf, params_df, Xsf, Prior_sf):

    p0 = params_sf.flatten()

    for component in ['xsf', 'xdf', 'integral', 'prior_sf', 'prior_df', 'full']:
        print(component)
        grad_anc = calc_nlnP_SFOnly(p0, Xsf, Prior_sf, params_df, component=component)[1].reshape(-1,6)
        like = lambda params: calc_nlnP_SFOnly(params, Xsf, Prior_sf, params_df, component=component, get_grad=False)
        grad_app = op.approx_fprime(p0, like, 1e-6).reshape(-1, 6)
        for i in range(grad_anc.shape[0]):
            print(grad_anc[i])
            print(grad_app[i])
            print("")
        print("")

def gradtest_x(params, df_params, m_i, xgrad_mu, xgrad_s, xgrad_pi,
           p, e_alpha_sf, p_pi, e_alpha_pi, Sdet_sf, cov):

    nlnP = - np.sum(np.log(m_i))
    grad = np.zeros((params.shape[0],6))
    grad[:,:2] = xgrad_mu
    grad[:,2] = xgrad_s[:,0,0] + xgrad_s[:,0,1]*cov/(2*params[:,2])
    grad[:,3] = xgrad_s[:,1,1] + xgrad_s[:,0,1]*cov/(2*params[:,3])
    grad[:,4] = xgrad_s[:,1,0]*np.sqrt(params[:,2]*params[:,3])* 2*e_alpha_sf/(1+e_alpha_sf)**2
    grad[:,5] = xgrad_pi * 2*np.pi*np.sqrt(Sdet_sf) * p_pi**2 * e_alpha_pi

    grad_df = np.zeros((df_params.shape[0],6))

    grad = np.hstack((grad.flatten(), grad_df.flatten()))

    return -np.sum(np.log(m_i)), grad
def gradtest_I(params, df_params, I,
           Igrad_mu, Igrad_s, Igrad_pi,
           Igrad_mudf, Igrad_sdf, Igrad_pidf,
           p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov, cov_df):

    grad = np.zeros((params.shape[0],6))
    grad[:,:2] = Igrad_mu
    grad[:,2] = Igrad_s[:,0,0] + Igrad_s[:,0,1]*cov/(2*params[:,2])
    grad[:,3] = Igrad_s[:,1,1] + Igrad_s[:,0,1]*cov/(2*params[:,3])
    grad[:,4] = Igrad_s[:,0,1]*np.sqrt(params[:,2]*params[:,3])* 2*e_alpha/(1+e_alpha)**2
    grad[:,5] = Igrad_pi * 2*np.pi*np.sqrt(Sdet_sf) * p_pi**2 * e_alpha_pi

    grad_df = np.zeros((df_params.shape[0],6))
    grad_df[:,:2] = Igrad_mudf
    grad_df[:,2] = Igrad_sdf[:,0,0] + Igrad_sdf[:,0,1]*cov_df/(2*params_df[:,2])
    grad_df[:,3] = Igrad_sdf[:,1,1] + Igrad_sdf[:,0,1]*cov_df/(2*params_df[:,3])
    grad_df[:,4] = Igrad_sdf[:,0,1]*np.sqrt(df_params[:,2]*df_params[:,3])
    grad_df[:,5] = Igrad_pidf

    grad = np.hstack((grad.flatten(), grad_df.flatten()))

    return np.sum(I), grad
def gradtest_prior(params, df_params, Prior,
           prgrad_mu, prgrad_s, prgrad_pi,
           p, e_alpha, p_pi, e_alpha_pi, Sdet_sf, cov):

    grad = np.zeros((params.shape[0],6))
    grad[:,:2] = prgrad_mu
    grad[:,2] = prgrad_s[:,0,0] + prgrad_s[:,0,1]*cov/(2*params[:,2])
    grad[:,3] = prgrad_s[:,1,1] + prgrad_s[:,0,1]*cov/(2*params[:,3])
    grad[:,4] = prgrad_s[:,0,1]*np.sqrt(params[:,2]*params[:,3])* 2*e_alpha/(1+e_alpha)**2
    grad[:,5] = prgrad_pi * 2*np.pi*np.sqrt(Sdet_sf) * p_pi**2 * e_alpha_pi

    grad_df = np.zeros((df_params.shape[0],6))

    grad = np.hstack((grad.flatten(), grad_df.flatten()))

    return -Prior, grad
def gradtest_priordf(params, df_params, Prior_df,
           prgrad_mudf, prgrad_sdf, prgrad_pidf, prgrad_Ndf,
           p, e_alpha, p_pi, e_alpha_pi, N, pi_df, Sdet_sf, cov_df):

    grad = np.zeros((params.shape[0],6))

    grad_df = np.zeros((df_params.shape[0],6))
    grad_df[:,:2] = prgrad_mudf
    grad_df[:,2] = prgrad_sdf[:,0,0] + prgrad_sdf[:,0,1]*cov_df/(2*df_params[:,2])
    grad_df[:,3] = prgrad_sdf[:,1,1] + prgrad_sdf[:,0,1]*cov_df/(2*df_params[:,3])
    grad_df[:,4] = prgrad_sdf[:,0,1]*np.sqrt(df_params[:,2]*df_params[:,3])
    grad_df[:,5] = prgrad_Ndf + prgrad_pidf

    grad = np.hstack((grad.flatten(), grad_df.flatten()))

    return -(Prior_df), grad
def gradtest_xdf(params, df_params, m_i_df,
           xgrad_mudf, xgrad_sdf, xgrad_pidf, xgrad_Ndf,
           p, e_alpha, p_pi, e_alpha_pi, N, pi_df, Sdet_sf, cov_df):

    grad = np.zeros((params.shape[0],6))

    grad_df = np.zeros((df_params.shape[0],6))
    grad_df[:,:2] = xgrad_mudf
    grad_df[:,2] = xgrad_sdf[:,0,0] + xgrad_sdf[:,0,1]*cov_df/(2*df_params[:,2])
    grad_df[:,3] = xgrad_sdf[:,1,1] + xgrad_sdf[:,0,1]*cov_df/(2*df_params[:,3])
    grad_df[:,4] = xgrad_sdf[:,0,1]*np.sqrt(df_params[:,2]*df_params[:,3])
    grad_df[:,5] = xgrad_pidf

    grad = np.hstack((grad.flatten(), grad_df.flatten()))

    return -np.sum(np.log(m_i_df)), grad


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

    def __init__(self, value=0., rangex=(0.,0.), rangey=(0.,0.), runscaling=False):

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
