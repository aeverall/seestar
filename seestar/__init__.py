__all__ = ['IsochroneScaling', 'ArrayMechanics', 'StatisticalModels',
 			'createNew', 'setdatalocation', 'DistributionPlots', 
 			'FieldUnions', 'SelectionGrid', 'FieldAssignment'
 			'SFInstanceClasses', 'surveyInfoPickler']

from sys import version as pyversion
if pyversion > '3': # Load in Python 3 version
    __version__ = 'v1.3.5-c'
elif (pyversion<'3') & (pyversion>'2.7'): # Load in Python 2.7 version
    __version__ = 'v1.2.7-c'