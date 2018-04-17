# File containing examples on how to use Pandas

#%% Import packages

import numpy as np
import pandas as pd
import os, sys

#%% Relative location of repository

directory = '[directory]'

#%% Load/create a pandas dataframe

file_path = os.path.join(directory, 'examples/ex_intrinsicdata.txt')
array = np.loadtxt(file_path)

dataframe = pd.DataFrame(array, columns=['glon', 'glat', 's', 'age', 'mh', 'mass'])

#%% Save dataframe to a comma-separated csv file

new_file_path = os.path.join(directory, 'examples/ex_intrinsicdata.csv')
dataframe.to_csv(new_file_path, index=False, header=True)


#%% Accessing components of data

# Access an individual column
glon = dataframe.glon
glon = dataframe['glon']

# Access a row (counted from top) of dataframe - starts at 0
row3 = dataframe.iloc[2]

# Access range of rows
somerows = dataframe.iloc[10:20]

# Access rows by index of row - same as iloc for this dataframe as index not changed
indexrowi = dataframe.loc[2]

# Set index to a column
dataframe = dataframe.set_index(dataframe.mass, drop=True)

# Find value on index
i = dataframe.mass.iloc[10]
row_massi = dataframe.loc[i]

# Reset the index to arbitrary integers
dataframe.reset_index(drop=True, inplace=True)

# Get sub-dataframe with specific columns
subsf = dataframe[['glon', 'glat']]
