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
import itertools, functools
import numpy as np
import pandas as pd
import sys, gc

from seestar import ArrayMechanics
from seestar import FieldAssignment

class FieldUnion():

    def __init__(self):

        pass

    def __call__(self, series):

        listofarr = self.ProbMatrix(series)

        combos = self.fieldCombos(listofarr)

        # Calculate the intersection of each field combo from the overlapping fields
        union_contributions = list(map(self.fieldIntersection, combos))

        # Find the sum of contributions from each combination intersection to the total union
        union = sum(union_contributions)

        return union

    def ProbMatrix(self, series):

        # Convert series to list
        lst = list(series)

        # Length of longest element
        length = len(sorted(lst,key=len, reverse=True)[0])

        # Extend all components to the right length
        arr=np.array([xi+[0]*(length-len(xi)) for xi in lst])
        
        # Convert to list of vectors
        listofarr=[arr[:,i] for i in range(arr.shape[1])]

        return listofarr

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

    def fieldIntersection(self, listofarr):

        '''
        fieldIntersection - Calculate the intersection of any group of fields

        Parameters
        ----------
            list_of_tuples - list of tuples of floats
                    - list of tuples of SFprob, fieldID for each field which is in the
                    particular combination

        Returns
        -------
            product*sign - float
                    - contribution to the field union from this field intersection
                    out of all of the combos
                    - contributions of all combos are summed in fieldUnion
        '''

        # Calculate the product of all probabilities
        #sf_value = [tup[0] for tup in list_of_tuples]
        product = functools.reduce(lambda x,y: x*y, listofarr)

        # Include correct sign for the union calculation
        sign = (-1)**(len(listofarr     )+1)

        return product*sign


def GenerateMatrices(df, pointings, angle_coords, point_coords, halfangle, SFcalc, 
                    IDtype = str, Nsample = 10000, test=False, progress=True):

    '''
    AnglePointsToPointingsMatrix - Adds a column to the df with the number of the field pointing
                                 - Uses matrix algebra
                                    - Fastest method for asigning field pointings
                                    - Requires high memory usage to temporarily hold matrices

    Parameters
    ----------
        df: pd.DataFrame
            Contains Theta and Phi column corresponding to the coordinates of points on the contingent axes (RA,Dec)

        pointings: pd.DataFrame
            Contains an x, y, and r column corresponding to positions and radii of field pointings

        angle_coords: tuple of str
            - Names of angle column headers in df
        point_coords: tuple of str
            - Names of angle column headers in pointings

        halfangle: string
            - Column header for half-angle of plate on sky

        SFcalc: lambda/function
            - Function for calculating probability of star being selected given field and coords

    kwargs
    ------
        IDtype: object
            - Type of python object used for field IDs 

        Nsample: int
            - Number of stars to be assigned per iterations
            Can't do too many at once due to computer memory constraints

        basis='intrinsic': str
            - (I don't think this is actually used)

    Returns
    -------
        df: pd.DataFrame
            Same as input df with:
                - 'points': list of field IDs for fields which the coordinates lie on
                - 'field_info': list of tuples - (P(S|v), fieldID) - (float, fieldIDtype)
    '''
    Nsample = FieldAssignment.iterLimit(len(pointings))

    pointings.rename(index=str, columns=dict(zip(point_coords, angle_coords)), inplace=True)
    df = ArrayMechanics.AnglePointsToPointingsMatrix(df, pointings, angle_coords[0], angle_coords[1], halfangle,
                                                        IDtype = IDtype, Nsample=Nsample, progress=progress)
    print("")

    iterated=0

    # Iterate over portions of size, Nsample to constrain memory usage.
    for it in range(int(len(df)/Nsample) + 1):

        dfi = df.iloc[it*Nsample:(it+1)*Nsample].copy()

        iterated +=  len(dfi)
        if progress: sys.stdout.write("\rCalculating: "+str(iterated)+'/'+str(len(df))+"        ")

        # Dataframe of field probabilities
        dfprob = pd.DataFrame()
        for field in pointings.fieldID:
        	
            # Condition: Boolean series - field is in the points list
            condition = np.array(dfi.points.map(lambda points: field in points))
            # Create array for probability values
            array = np.zeros(len(dfi)) - 1
            # Calculate probabilities
            if test: 
                prob, col, mag = SFcalc(field, dfi[condition])
                col_arr = np.zeros(len(dfi)) - 1
                mag_arr = np.zeros(len(dfi)) - 1
                col_arr[condition] = col
                mag_arr[condition] = mag
            else: prob = SFcalc(field, dfi[condition])
            # Set probability values in array
            if isinstance(prob, pd.Series):
                array[condition] = prob.values
            else: array[condition] = prob
            # Add column to dfprob dataframe
            dfprob[field] = array
        # dfprob now has a column for every field with pvalues (or -1s)

        if test:
            dfi['col'] = col_arr
            dfi['mag'] = mag_arr

        # Remove -1 entries from the lists
        def filtering(x, remove):
            x = [elem for elem in x if elem!=remove]
            return x
        # Convert SFprob values into list of values in dataframe
        arr = np.array(dfprob)
        # Do filtering for fields
        listoflists = arr.tolist()
        listoflists = [filtering(x, -1) for x in listoflists]

        # Lists of SF probabilities
        SFprob = pd.DataFrame(pd.Series(listoflists), columns=['SFprob'])

        # zip datatypes together - tupes of (sf, field)
        field_info = [list(zip(SFprob.SFprob.iloc[i], dfi.points.iloc[i])) for i in range(len(dfi))]
        field_info = pd.DataFrame(pd.Series(field_info), columns=['field_info'])

        # Reset index to merge on position then bring index back
        if 'index' in list(dfi): dfi.drop('index', axis=1, inplace=True)
        dfi.reset_index(inplace=True)
        dfi = dfi.merge(SFprob, how='inner', right_index=True, left_index=True)
        dfi = dfi.merge(field_info, how='inner', right_index=True, left_index=True)
        dfi.index = dfi['index']
        dfi.drop('index', axis=1, inplace=True)

        if test:
            dfi['col'] = col_arr
            dfi['mag'] = mag_arr
            return dfi, col_arr, mag_arr

        if it == 0: newdf = dfi
        else: newdf = pd.concat((newdf, dfi))

    return newdf