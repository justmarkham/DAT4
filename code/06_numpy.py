'''
Numpy Reference Guide

Sources:
    http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf
    Book: Python for Data Analysis (Chapter 4)
'''

import numpy as np
import matplotlib.pyplot as plt

# create ndarrays from lists
# note: every element must be the same type (will be converted if possible)
data1 = [6, 7.5, 8, 0, 1]           # list
arr1 = np.array(data1)              # 1d array
data2 = [range(1, 5), range(5, 9)]  # list of lists
arr2 = np.array(data2)              # 2d array
arr2.tolist()                       # convert array back to list

# examining arrays
arr1.dtype      # float64
arr2.dtype      # int32
arr2.ndim       # 2
arr2.shape      # (2, 4) - axis 0 is rows, axis 1 is columns
arr2.size       # 8 - total number of elements
len(arr2)       # 2 - size of first dimension (aka axis)

# create special arrays
np.zeros(10)
np.zeros((3, 6))
np.ones(10)
np.linspace(0, 1, 5)            # 0 to 1 (inclusive) with 5 points
np.logspace(0, 3, 4)            # 10^0 to 10^3 (inclusive) with 4 points

# arange is like range, except it returns an array (not a list)
int_array = np.arange(5)
float_array = int_array.astype(float)

# slicing
arr1[0]         # 0th element (slices like a list)
arr2[0]         # row 0: returns 1d array ([1, 2, 3, 4])
arr2[0, 3]      # row 0, column 3: returns 4
arr2[0][3]      # alternative syntax
arr2[:, 0]      # all rows, column 0: returns 1d array ([1, 5])
arr2[:, 0:1]    # all rows, column 0: returns 2d array ([[1], [5]])

# views and copies
arr = np.arange(10)
arr[5:8]                    # returns [5, 6, 7]
arr[5:8] = 12               # all three values are overwritten (would give error on a list)
arr_view = arr[5:8]         # creates a "view" on arr, not a copy
arr_view[:] = 13            # modifies arr_view AND arr
arr_copy = arr[5:8].copy()  # makes a copy instead
arr_copy[:] = 14            # only modifies arr_copy

# using boolean arrays
names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
names == 'Bob'                          # returns a boolean array
names[names != 'Bob']                   # logical selection
(names == 'Bob') | (names == 'Will')    # keywords "and/or" don't work with boolean arrays
names[names != 'Bob'] = 'Joe'           # assign based on a logical selection
np.unique(names)                        # set function

# vectorized operations
nums = np.arange(5)
nums*10                             # multiply each element by 10
nums = np.sqrt(nums)                # square root of each element
np.ceil(nums)                       # also floor, rint (round to nearest int)
np.isnan(nums)                      # checks for NaN
nums + np.arange(5)                 # add element-wise
np.maximum(nums, np.array([1, -2, 3, -4, 5]))  # compare element-wise


# math and stats
rnd = np.random.randn(4, 2) # random normals in 4x2 array
rnd.mean()
rnd.std()
rnd.argmin()                # index of minimum element
rnd.sum()
rnd.sum(axis=0)             # sum of columns
rnd.sum(axis=1)             # sum of rows


# use numpy to create scatter plots
N = 50

x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area =30+(70*np.random.rand(N)) # 30 to 100 point radiuses

plt.scatter(x, y, s=area, c=colors,)
plt.show()


# conditional logic
np.where(rnd > 0, 2, -2)    # args: condition, value if True, value if False
np.where(rnd > 0, 2, rnd)   # any of the 3 arguments can be an array

# methods for boolean arrays
(rnd > 0).sum()             # counts number of positive values
(rnd > 0).any()             # checks if any value is True
(rnd > 0).all()             # checks if all values are True

# reshape, transpose, flatten
nums = np.arange(32).reshape(8, 4) # creates 8x4 array
nums.T                       # transpose
nums.flatten()               # flatten


# random numbers
np.random.seed(12234)
np.random.rand(2, 3)      # 0 to 1, in the given shape
np.random.randn(10)         # random normals (mean 0, sd 1)
np.random.randint(0, 2, 10) # 0 or 1