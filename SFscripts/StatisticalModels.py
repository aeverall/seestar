
'''
StatisticalModels.py - Set of functions for building statistical models, calculations and tests.

Parameters
----------


**kwargs
--------


Returns
-------


'''

import numpy as np
import numpy
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.optimize as op
import sys
from mpmath import *
import Optimizers

# Import cubature for integrating over regions
from cubature import cubature

def bivariateGauss(params, x, y):
    mu1, sigma1, mu2, sigma2, A, rho = params
    # Coordinate
    z = ((x - mu1)**2 / sigma1**2) + ((y - mu2)**2 / sigma2**2) + \
        2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
    # When the Series is empty, the data type goes to object so this is corrected:
    z = z.astype(np.float64)
    # Bivariate Gaussian
    Norm = (A/(2 * np.pi * np.abs(sigma1 * sigma2) * np.sqrt(1 - rho**2)))
    Exponent = numpy.exp(-z / (2 * (1 - rho**2)))
    BG = Norm*Exponent
    return BG

def bivariateIntegral(params):
    mux, sigmax, muy, sigmay, A, rho = params
    # Continuous integral of Bivariate Gaussian with infinite boundaries.
    contInteg = 2*np.pi * A * np.abs(sigmax * sigmay) * np.sqrt(1-rho**2)
    return contInteg

def multiDistribution(params, x, y, nComponents):
	p = 0
	for i in range(nComponents):
		p += bivariateGauss(params[i], x, y)
	return p
def multiIntegral(params, nComponents):
	integral = 0
	for i in range(nComponents):
		integral += bivariateIntegral(params[i])
	return integral
    
# Gaussian function for generating error distributions
def Gauss(x, mu=0, sigma=1):
    G = np.exp(-((x-mu)**2)/(2*sigma**2))
    return G

# Create cumulative distribution from Gauss(x)
def cdf(func, xmin, xmax, N, **kwargs):
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
    value_interp = interp.interp1d(cdf, points[1:])
    
    return value_interp



"""
# ln(Likelihood) based on a Poisson likelihood distribution
def lnlike(params, x, y):
    vars1 = params[:6]
    vars2 = params[6:12]
    # For scipy.optimise - lnprior is not used so rho must be forced within bounds
    if vars1[5]**2 < 1 and vars2[5]**2 < 1:
        model = Double([vars1, vars2], x, y)
        contPoints = np.sum( np.log(model) )
        if np.isnan(contPoints):
            print(model)
        # Integral over region for 2D Gaussian distribution
        contInteg1 = bivariateIntegral(vars1)
        contInteg2 = bivariateIntegral(vars2)
        logl =  contPoints + contInteg1 + contInteg2
        if np.isnan(logl):
            print(contPoints, contInteg1, contInteg2, params)
        return logl
    else: return -np.inf

# "uninformative prior" - uniform and non-zero within a specified range of parameter values
def lnprior(params):
    # For now we'll just neglect the possibility of a prior
    noprior = True
    if noprior:
        return 0.0
    return -np.inf

# posterior probability function is proportional to the prior times the likelihood
# lnpost = lnprior + lnlike
def lnprob(params, x, y):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, x, y)
"""


def GMMParameters(nComponents, rngx, rngy):

    # Initial guess parameters for a bivariate Gaussian
    mux_i, muy_i = (rngx[0]+rngx[1])/2, (rngy[0]+rngy[1])/2
    sigmax_i, sigmay_i = 0.5, 0.5
    A_i = 10.
    rho_i = 0.
    p_list = [mux_i, sigmax_i, muy_i, sigmay_i, A_i, rho_i]

    mux_u, muy_u = rngx[0], rngy[0]
    sigmax_u, sigmay_u = 0, 0
    A_u = 0.
    rho_u = -1.
    u_list = [mux_u, sigmax_u, muy_u, sigmay_u, A_u, rho_u]

    mux_o, muy_o = rngx[1], rngy[1]
    sigmax_o, sigmay_o = rngx[1]-rngx[0], rngy[1]-rngy[0]
    A_o = np.inf
    rho_o = 1.
    o_list = [mux_o, sigmax_o, muy_o, sigmay_o, A_o, rho_o]

    # Initial parameters for a Double bivariate Gaussian
    parameters_i = [p_list,]*nComponents
    parameters_u = [u_list,]*nComponents
    parameters_o = [o_list,]*nComponents

    return parameters_i, parameters_u, parameters_o

def numericalIntegrate(function, (rngx, rngy)):
    
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

    #integral, err = cubature(function, 2, 1, (rngx[0], rngy[0]), (rngx[1], rngy[1]))

    #error = integral - integral1
    #print(error)
    return integral

def simpsonIntegrate(function, (rngx, rngy)):

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



class GaussianMM():
    
    def __init__(self, x, y, nComponents, rngx, rngy):
        
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

        self.params_i, self.underPriors, self.overPriors = GMMParameters(nComponents, rngx, rngy)
        
        self.x = x
        self.y = y

        self.rngx, self.rngy = rngx, rngy
        
        # Function which calculates the actual distribution
        self.distribution = multiDistribution

        #Print out values of L as calculated
        self.runningL = False
        
    def __call__(self, (x, y)):
        
        return self.distribution(self.params_f, x, y, self.nComponents)
        
        
    def optimizeParams(self, method = "SLSQP"):

        # nll is the negative lnlike distribution
        nll = lambda *args: -self.lnprob(*args)

        # result is the set of theta parameters which optimise the likelihood given x, y, yerr
        result = op.minimize(nll, self.params_i, method = method)
        
        # Save evaluated parameters to internal values
        self.params_f = []
        for i in range(self.nComponents):
            self.params_f.append(result["x"][i*6:(i+1)*6])
       
        return nll(result["x"])

    # ln(Likelihood) based on a Poisson likelihood distribution
    def lnlike(self, params):
        
        param_set = []
        for i in range(self.nComponents):
            param_set.append( params[i*6:(i+1)*6] )

        model = self.distribution(param_set, self.x, self.y, self.nComponents)
        contPoints = np.sum( np.log(model) )
            
        # Integral over region for 2D Gaussian distribution
        function = lambda (a, b): self.distribution(param_set, a, b, self.nComponents)
        
        # Decide the type of integral we want to conduct for the region.
        integration = "trapezium"
        # analytic if we have analytic solution to the distribution - this is the fastest
        if integration == "analytic": contInteg = multiIntegral(param_set, self.nComponents)
        # trapezium is a simple approximation for the integral - fast - ~1% accurate
        elif integration == "trapezium": contInteg = numericalIntegrate(function, (self.rngx, self.rngy))
        # simpson is a quadratic approximation to the integral - reasonably fast - ~1% accurate
        elif integration == "simpson": contInteg = simpsonIntegrate(function, (self.rngx, self.rngy))
        # cubature is another possibility but this is far slower!
        elif integration == "cubature": 
            contInteg, err = cubature(func2d, 2, 1, (self.rngx[0], self.rngy[0]), (self.rngx[1], self.rngy[1]))
            contInteg = float(contInteg)

        lnL = contPoints - contInteg

        if self.runningL:
            print(self.runningL)
            sys.stdout.write("\rlogL: %.2f, sum log(f(xi)): %.2f, integral: %.2f" % (lnL, contPoints, contInteg))
            sys.stdout.flush()

            
        return contPoints - contInteg

    # "uninformative prior" - uniform and non-zero within a specified range of parameter values
    def lnprior(self, params):
        
        param_set = []
        for i in range(self.nComponents):
            param_set.append( params[i*6:(i+1)*6] )
        prior = self.priorTest(param_set)

        if prior:
            return 0.0
        else: 
			return -np.inf

    # posterior probability function is proportional to the prior times the likelihood
    # lnpost = lnprior + lnlike
    def lnprob(self, params):
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(params)
    
    # Returns a boolean true or false for whether all parameters lie within their range
    def priorTest(self, params):
        
        Val = np.array(params).flatten()
        minVal = np.array(self.underPriors).flatten()
        maxVal = np.array(self.overPriors).flatten()
        
        minBool = Val > minVal
        maxBool = Val < maxVal
        rngBool = minBool*maxBool
        
        solution = np.sum(rngBool) - len(Val)
        if solution == 0:
            return True
        else: return False

    def testIntegral(self):

        function = lambda (a, b): self.distribution(self.params_f, a, b, self.nComponents)

        real_val, err = cubature(function, 2, 1, (self.rngx[0], self.rngy[0]), (self.rngx[1], self.rngy[1]))
        calc_val = simpsonIntegrate(function, (self.rngx, self.rngy))

        percent = ((calc_val - float(real_val))/calc_val)*100
        cubature_percent = 100*float(err)/float(real_val)

        print("\nThe error in the linear numerical integral was %.3E%%" % float(percent))
        print("\nThe cubature calculation error is quoted as %.3E or %.3E%%" % (float(err), cubature_percent)  )

        return calc_val, real_val, err

'''

Functions for penalised grid fitting models

'''


def gridDistribution((nx, ny), (rngx, rngy), params):
    
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
    
    #compInteg = integ.dblquad(function, rngx[0], rngx[1], rngy[0], rngy[1])
    compInteg, err = cubature(function, 2, 1, (rngx[0], rngy[0]), (rngx[1], rngy[1]))
    
    return compInteg

class PenalisedGridModel():
    
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
        
        # Final interpolant is set after generateInterpolant is called
        
        return self.finalInterpolant((xi, yi))
        
    def generateInterpolant(self):
        
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
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(params)
    
    # Returns a boolean true or false for whether all parameters lie within their range
    def priorTest(self, params):
        
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