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


def pickleBoundMethods():

	def _pickle_method(method):
		func_name = method.im_func.__name__
		obj = method.im_self
		cls = method.im_class
		if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
			cls_name = cls.__name__.lstrip('_')
			func_name = '_' + cls_name + func_name
		return _unpickle_method, (func_name, obj, cls)

	def _unpickle_method(func_name, obj, cls):
		for cls in cls.__mro__:
			try:
				func = cls.__dict__[func_name]
			except KeyError:
				pass
			else:
				break
		return func.__get__(obj, cls)

	import copy_reg
	import types
	copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

	print("Pickle bound methods activated.")