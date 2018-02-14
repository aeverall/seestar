'''
FieldUnions - Package of functions used to calculate the union of overlapping fields given the 
			coordinates of a star which has been randomly generated.


Classes
-------
    FieldInterpolants - Class for building a dictionary of interpolants for each of the survey Fields
                        The interpolants are used for creating Field selection functions

Functions
---------
	FieldToTuple - Converts from a field and coordinates on the field to
				   a selection probability and angular coordinates of the field.

	pointsListMap - Runs FieldToTuple for every item in the list of fields which are overlapping
					on the point.

	fieldUnion - The field union on a point of overlapping fields with selection function
				 for each field already calculated									 

	fieldCombos - Generate list of combinations of the overlapping fields
				  which will be used to calculate the intersections.

	fieldIntersection - Calculate the intersection of any group of fields

	FieldOverlap - Calculation of the fraction of N fields which are overlapping.

Requirements
------------

'''
import itertools


def FieldToTuple(selectionfunction, fieldID, (age, mh, s)):

 	'''
	FieldToTuple - Converts from a field and coordinates on the field to
				   a selection probability and angular coordinates of the field.

 	Parameters
 	----------
 		selectionfunction - Dictionary/DataFrame
 					- The full distribution of selection functions in age-mh-s space for each field

 		fieldID - object type depends on survey
 					- The field label

 		(age, mh, s) - tuple of floats
					- Parameters of the star for which we're calculating the selection function probability

 	Returns
 	-------
 		sf - float
 				 - Value of the selection function probability for these coordinates

 		info.l, info.b - floats
 				- Galactic longitude & latitude of the field
 	'''
    
	info = selectionfunction.loc[fieldID]

	# selection function value
	interp = info.agemhssf
	try:
	    sf = interp((age, mh, s))
	except IndexError: 
	    sf=0.
	    
	return sf, info.l, info.b



def pointsListMap(row, sf):

	'''
	pointsListMap - Runs FieldToTuple for every item in the list of fields which are overlapping
					on the point.

	Parameters
	----------
		row - row from DataFrame

		row["points"] - list
			- list of fields which overlap on a given point in Galactic/Equatorial coordinates.

		sf - dictionary of interpolants
			- self.surveysf from the selection function which can be used to determine the selection
			 probability given those parameters on that field.

	Returns
	-------
		list of tuples
			- for every field in row["points"], the (SFprobability, Glongitude, Glatitude)
	'''

	return [FieldToTuple(sf, item, (row['age'], row['mh'], row['s'])) for item in row['points']]

def fieldUnion(field_info):

	'''
	fieldUnion - The field union on a point of overlapping fields with selection function
				 for each field already calculated

	Parameters
	----------
		field_info - list of tuples
			- tuples contain SFvalue and galactic coordinates of field

	Returns
	-------
		union - float
			- Final value of selection function for given star.
	'''

	# Creates a list of all combinations of fields from the overlapping fields.
	# A sum of the intersections of these fields (x-1^i) gives the field union.
	combos = fieldCombos(field_info)

	# Calculate the intersection of each field combo from the overlapping fields
	union_contributions = map(fieldIntersection, combos)

	# Find the sum of contributions from each combination intersection to the total union
	union = sum(union_contributions)

	return union

def fieldCombos(field_info):

	'''
	fieldCombos - Generate list of combinations of the overlapping fields
				  which will be used to calculate the intersections.

	Parameters
	----------
		field_info - list of tuples
				- Each tuple has the SFprob, l and b of the field

	Returns
	-------
		list of lists of combos of tuples
	'''

	combos = []
	for i in range(len(field_info)):
		combos.extend(list(itertools.combinations(field_info, i+1)))
	    
	return combos

def fieldIntersection(list_of_tuples):

	'''
	fieldIntersection - Calculate the intersection of any group of fields

	Parameters
	----------
		list_of_tuples - list of tuples of floats
				- list of tuples of SFprob, l, b for each field which is in the
				particular combination

	Returns
	-------
		product*sign - float
				- contribution to the field union from this field intersection
				out of all of the combos
				- contributions of all combos are summed in fieldUnion
	'''

	# Calculate the product of all probabilities
	sf_value = map(lambda tup: tup[0], list_of_tuples)
	product = reduce(lambda x,y: x*y, sf_value)

	# Calculate the fraction of overlap between fields
	l = map(lambda tup: tup[1], list_of_tuples)
	b = map(lambda tup: tup[2], list_of_tuples)
	f = FractionOverlap(l, b)
	# Currently this just returns 1 but we need to calculate this

	# Include correct sign for the union calculation
	sign = (-1)**(len(list_of_tuples)+1)

	return product*sign

def FractionOverlap(l, b):

	'''
	FieldOverlap - Calculation of the fraction of N fields which are overlapping.

	Parameters
	----------
		l, b - list of floats
			- Galactic coordinates of fields for which we're calculating the overlap

		SA - float
			- The solid angle of the fields 

	Returns
	-------
		ratio - float
			- The ratio of the overlap area of the fields to the total area of both fields
	'''
	ratio = 1

	return ratio