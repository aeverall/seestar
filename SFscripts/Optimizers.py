import math
import StatisticalModels
import numpy as np
import sys

class nonMarkovOptimizer():

	def __init__(self, func, pInit):

		# Number of iterations in relaxation phase
		self.nrel = 100
		# Number of iterations in logarithmic phase
		self.nlog = 1000

		# Divisor or multiplier for sigma in order to converge result
		self.uConverge = 1.01
		# Initial parameters for distribution
		self.pInit = pInit
		# Likelihood function which we are attempting to optimize
		self.func = func

		# Gaussian standard deviation for shifting parameter value
		self.sigmaInit = 1.
		# Boundaries used to determine the cumulative distribution function
		# for probability weighted selection of next parameter
		self.cdfBounds = (0., 100)

		# List of values of F calculated by varing parameters
		self.F = []

	def __call__(self):

		self.pCurrent = self.pInit
		self.sigma = self.sigmaInit

		self.relaxationPhase()

		print("relaxed")

		#while self.sigma>0.1:
		for count in range(self.nlog):
			self.improvementPhase()
			#sys.stdout.write("\rsigma: %.2f" % (self.sigma))
			#sys.stdout.flush()

		output = {}
		output["x"] = self.pCurrent

		return output


	def proposalF(self):

		# Randomly select parameter to be varied.
		#iTot, jTot = np.shape(pInit)
		#iIndex = math.floor((np.random.rand() * iTot))
		#jIndex = math.floor((np.random.rand() * jTot))
		index = int( math.floor((np.random.rand() * len(self.pInit))) )

		# Select next parameter value
		nextValDist = lambda x: StatisticalModels.Gauss(x, mu=self.pCurrent[index], sigma=self.sigma)
		nextValCDF = StatisticalModels.cdf(nextValDist, self.cdfBounds[0], self.cdfBounds[1], 1000)
		proposedVal = nextValCDF( np.random.rand() )
		proposedParams = self.pCurrent
		#proposedParams[iIndex, jIndex] = proposedVal
		proposedParams[index] = proposedVal

		# function result for this new value
		F = self.func(proposedParams)
		self.F.append(F)

		return proposedParams

	def relaxationPhase(self):

		for i in range(self.nrel):
			self.pCurrent = self.proposalF()


	def improvementPhase(self):

		proposedParams = self.proposalF()

		meanDF = np.sum( np.array( self.F[:-1] ) - np.array( self.F[1:] ) ) / (len(self.F) - 1)
		DF = self.F[-1] - self.F[-2]

		alpha = np.exp(DF/meanDF)

		if alpha >= np.random.rand():
			self.pCurrent = proposedParams
			#self.sigma /= self.uConverge
		else:
			self.F = self.F[:-1]
			#self.sigma *= self.uConverge