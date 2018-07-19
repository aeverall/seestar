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
        print(len(combos), combos[5])

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

    def pointsListMap(self, sf, row):

        '''
        NO LONGER USED
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


    def FieldToTuple(self, selectionfunction, fieldID, age, mh, s):
        '''
        NO LONGER USED
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

        sys.stdout.write("\r"+str(sf)+'...'+str(fieldID))
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
        sign = (-1)**(len(listofarr)+1)

        return product*sign




# Calculating field fractional overlaps

def GenerateRandomPoints(N, labels=['phi', 'theta', 'halfangle']):

    # Latitude weighted by cosine angle distribution
    brand = np.arcsin(2*np.random.rand(N)-1)
    lrand = np.random.rand(N)*2*np.pi
    # Generate dataframe for random points across sky
    coordsrand = np.vstack((lrand, brand)).T
    coordsrand = pd.DataFrame(coordsrand, columns=[labels[0], labels[1]])
    
    return coordsrand

def CalculateIntersections(coordsrand, fields, IDtype,  labels=['phi','theta','halfangle']):

    print('Counting intersection area fractions...')

    # Add column with the list of fields which overlap at each point
    fields = ArrayMechanics.AnglePointsToPointingsMatrix(coordsrand, fields, labels[0], labels[1], labels[2], IDtype=IDtype)

    # Only consider points with field
    intersections = fields[fields.points.map(len)>0].copy()

    intersections['strings'] = fields.points.astype(str)
    # Groupby field overlaps and count instances to measure area of region
    countseries = intersections.groupby(["strings"]).size()
    
    # Drop duplicates from intersections in order to merge with countseries 
    intersections = intersections.drop_duplicates(subset='strings')

    # Merge count series with intersection
    intersections = pd.merge(intersections, pd.DataFrame(countseries, columns=['counts']), left_on="strings", right_index=True)

    # Reset index to keep track of progress
    intersections = intersections.reset_index(drop=True)
    intersections['nIndex'] = intersections.index.astype(int)
    nfields = len(intersections)

    def findregions(fieldid, nIter):
        regionbool = intersections.apply(lambda row: any(i in row.points for i in fieldid), axis=1)
        # This can take a while so show counts as it goes along.
        sys.stdout.write("%d / %d\r" % (nIter+1, nfields))
        sys.stdout.flush()
        return np.sum(intersections.counts[regionbool])
    # For each overlap list, find the total area occupied by overlapping fields
    intersections['fullregion'] = intersections.apply(lambda row: findregions(row.points, row.nIndex), axis=1)
    
    # Intersection fraction is the ratio of counts on region to counts across all overlapping fields
    intersections['fraction'] = intersections.counts/intersections.fullregion
    
    print("\n...done")

    intersections = intersections.set_index("strings")
    
    return intersections[["points", "counts", "fullregion", "fraction"]] 
    
def CreateIntersectionDatabase(N, fields, IDtype, labels=['phi', 'theta', 'halfangle']):
    
    # Generate full set of points over the entire sky
    coordsrand = GenerateRandomPoints(N, labels=labels)
    # Calculate fraction overlaps for each field
    database = CalculateIntersections(coordsrand, fields, IDtype, labels=labels)
    
    return database


def GenerateMatrices(df, pointings, angle_coords, point_coords, halfangle, SFcalc, 
                    IDtype = str, Nsample = 10000, test=False):

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
                                                        IDtype = IDtype, Nsample=Nsample, progress=True)
    # Dataframe of field probabilities
    #arr = np.zeros((len(df), len(pointings))).astype(int) - 1 # -1 so that it's an impossible SFprob value
    #dfprob = pd.DataFrame(arr, columns=pointings.fieldID.tolist())
    dfprob = pd.DataFrame()

    for field in pointings.fieldID:
    	
        # Condition: Boolean series - field is in the points list
        condition = np.array(df.points.map(lambda points: field in points))
        # Create array for probability values
        array = np.zeros(len(df)) - 1
        # Calculate probabilities
        if test: 
            prob, col, mag = SFcalc(field, df[condition])
            col_arr = np.zeros(len(df)) - 1
            mag_arr = np.zeros(len(df)) - 1
            col_arr[condition] = col
            mag_arr[condition] = mag
        else: prob = SFcalc(field, df[condition])
        # Set probability values in array
        if isinstance(prob, pd.Series):
            array[condition] = prob.values
        else: array[condition] = prob
        # Add column to dfprob dataframe
        dfprob[field] = array
    # dfprob now has a column for every field with pvalues (or -1s)

    if test:
        df['col'] = col_arr
        df['mag'] = mag_arr

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
    field_info = [list(zip(SFprob.SFprob.iloc[i], df.points.iloc[i])) for i in range(len(df))]
    field_info = pd.DataFrame(pd.Series(field_info), columns=['field_info'])

    # Reset index to merge on position then bring index back
    if 'index' in list(df): df.drop('index', axis=1, inplace=True)
    df.reset_index(inplace=True)
    df = df.merge(SFprob, how='inner', right_index=True, left_index=True)
    df = df.merge(field_info, how='inner', right_index=True, left_index=True)
    df.index = df['index']
    df.drop('index', axis=1, inplace=True)

    if test:
        df['col'] = col_arr
        df['mag'] = mag_arr
        return df, col_arr, mag_arr

    return df