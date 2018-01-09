import cProfile
import pstats

class LogData():

	def start(self):

		self.pr = cProfile.Profile()
		self.pr.enable()

	def stop(self):

		self.pr.disable()

	def display(self):

		pstats.Stats(self.pr).strip_dirs().sort_stats('time').print_stats(10)