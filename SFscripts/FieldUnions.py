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
import numpy as np
import pandas as pd
import ArrayMechanics
import sys

class FieldUnion():

	def __init__(self, overlapdata):

		self.Overlaps = overlapdata
		

	def pointsListMap(self, sf, row):

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

		return [self.FieldToTuple(sf, item, (row['age'], row['mh'], row['s'])) for item in row.points]


	def FieldToTuple(self, selectionfunction, fieldID, (age, mh, s)):

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
		except (ValueError, IndexError): 
		    sf = 0.
		    
		return sf, fieldID


	def fieldUnion(self, field_info):

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
		combos = self.fieldCombos(field_info)

		# Calculate the intersection of each field combo from the overlapping fields
		union_contributions = map(self.fieldIntersection, combos)

		# Find the sum of contributions from each combination intersection to the total union
		union = sum(union_contributions)

		return union

	def fieldCombos(self, field_info):

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

	def fieldIntersection(self, list_of_tuples):

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
		fieldIDs = map(lambda tup: tup[1], list_of_tuples)
		f = self.FractionOverlap(fieldIDs)
		# Currently this just returns 1 but we need to calculate this

		# Include correct sign for the union calculation
		sign = (-1)**(len(list_of_tuples)+1)

		return product*sign

	def FractionOverlap(self, fieldIDs):

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

		try: ratio = self.Overlaps.loc[str(fieldIDs)].fraction
		except KeyError: ratio = 0.

		return ratio



# Calculating field fractional overlaps

def GenerateRandomPoints(N):

    # Latitude weighted by cosine angle distribution
    brand = np.arcsin(2*np.random.rand(N)-1)
    lrand = np.random.rand(N)*2*np.pi
    # Generate dataframe for random points across sky
    coordsrand = np.vstack((lrand, brand)).T
    coordsrand = pd.DataFrame(coordsrand, columns=["l", "b"])
    
    return coordsrand

def CalculateIntersections(coordsrand, fields, IDtype):

    print('Counting intersection area fractions...')

    # Add column with the list of fields which overlap at each point
    fields = ArrayMechanics.AnglePointsToPointingsMatrix(coordsrand, fields, "l", "b", 'SolidAngle', IDtype=IDtype)
    
    # Only consider points with field
    intersections = fields[fields.points.map(len)>0]
    # Column of string list required to group items
    print(intersections.points.iloc[0][0])
    print(type(intersections.points.iloc[0][0]))
    intersections['strings'] = fields.points.astype(str)
    # Groupby field overlaps and count instances to measure area of region
    countseries = intersections.groupby(["strings"]).size()
    
    # Drop duplicates from intersections in order to merge with countseries 
    intersections = intersections.drop_duplicates(subset='strings')

    # Merge count series with intersection
    intersections = pd.merge(intersections, pd.DataFrame(countseries, columns=['counts']), left_on="strings", right_index=True)

    # Reset index to keep track of progress
    intersections = intersections.reset_index(drop=True)
    
    def findregions(fieldid, index):
        regionbool = intersections.apply(lambda row: any(i in row.points for i in fieldid), axis=1)
        # This can take a while so show counts as it goes along.
        sys.stdout.write("%d / %d\r" % (index, len(intersections)))
        sys.stdout.flush()
        return np.sum(intersections.counts[regionbool])
    # For each overlap list, find the total area occupied by overlapping fields
    intersections['fullregion'] = intersections.apply(lambda row: findregions(row.points, row['index']), axis=1)
    
    # Intersection fraction is the ratio of counts on region to counts across all overlapping fields
    intersections['fraction'] = intersections.counts/intersections.fullregion
    
    print("...done")

    intersections = intersections.set_index("strings")
    
    return intersections[["points", "counts", "fullregion", "fraction"]] 
    
def CreateIntersectionDatabase(N, fields, IDtype):
    
    # Generate full set of points over the entire sky
    coordsrand = GenerateRandomPoints(N)
    # Calculate fraction overlaps for each field
    database = CalculateIntersections(coordsrand, fields, IDtype)
    
    return database