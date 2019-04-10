"""
Combine.py - Methods for combining selection function probabilites from different datasets.

Currently a lot of degeneracy with FieldUnions.py
I might replace FieldUnions with this or at least just leave FieldUnions for calculation of field-by-field probabilities.
(and maybe rename it to Field Probabilities.)
"""

import numpy as np, pandas as pd
import itertools, functools

def union_iterate(p_values):

    """
    union_iterate - compute the union of a set of probabilities.
        - Iterate through components (calculating the union pairwise)

    Parameters
    ----------
        p_values - 2d np.array of floats (n x d)
            - n - number of samples
            - d - number of probabilities per sample

    Returns
    -------
        p_union - np.array of floats (n)
            - n - number of samples

    The union of a set of probabilities is the same as iteratively calculating unions.
    A U B U C ... = ((A U B) U C)...
    Here we find it more efficient both in computations and memory allocation to perform iterations.
    """

    p_union = np.zeros(p_values.shape[0])
    for i in range(p_values.shape[1]):
        p_union = p_union+p_values[:,i] - (p_union*p_values[:,i])

    return p_union

def reshape_array(ll, fill_val=0):

    """
    reshape_array - converts a set of uneven lists to a n x d array.

    Parameters
    ----------
        ll - list/arr/series of lists of floats
            - Set of uneven lists.

    Returns
    -------
        arr - 2d np.array of floats (n x d)
            - n - number of samples
            - d - max length of a list in the samples
    """

    ll = list(ll)

    # Length of longest element
    length = max([len(l) for l in ll])
    # Extend all components to the right length
    arr=np.array([xi+[fill_val]*(length-len(xi)) for xi in ll])

    return arr

def union(series):

    """
    union - compute the union of a set of probabilities.

    Starting with how FieldUnions.py calculates this but I think it's highly inefficient.

    Parameters
    ----------
        p_values - 2d np.array of floats (n x d)
            - n - number of samples
            - d - number of probabilities per sample

    Returns
    -------
        p_union - np.array of floats (n)
            - n - number of samples
    """

    listofarr = ProbMatrix(series)

    combos = fieldCombos(listofarr)

    # Calculate the intersection of each field combo from the overlapping fields
    union_contributions = list(map(fieldIntersection, combos))

    # Find the sum of contributions from each combination intersection to the total union
    union = sum(union_contributions)

    return union

def ProbMatrix(series):

    # Convert series to list
    lst = list(series)

    # Length of longest element
    length = length = max([len(l) for l in lst])

    # Extend all components to the right length
    arr=np.array([xi+[0]*(length-len(xi)) for xi in lst])

    # Convert to list of vectors
    listofarr=[arr[:,i] for i in range(arr.shape[1])]

    return listofarr

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

def fieldIntersection(listofarr):

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
