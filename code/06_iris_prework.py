'''
EXERCISE: "Human Learning" with iris data
'''

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the famous iris data
iris = load_iris()

# what do you think these attributes represent?
iris.data
iris.data.shape
iris.feature_names
iris.target
iris.target_names

# intro to numpy
type(iris.data)


## PART 1: Read data into pandas and explore

# the feature_names are a bit messy, let's clean them up. remove the (cm)
# at the end and replace any spaces with an underscore
# create a list "features" that holds the cleaned column names

# read the iris data into pandas, with our refined column names

# create a list of species (should be 150 elements) 
# using iris.target and iris.target_names
# resulting list should only have the words "setosa", "versicolor", and "virginica"


# add the species list as a new DataFrame column


# explore data numerically, looking for differences between species
# try grouping by species and check out the different predictors


# explore data by sorting, looking for differences between species


# I used values in order to see all of the data at once
# without .values, a dataframe is returned


# explore data visually, looking for differences between species
# try using a histogram or boxplot





## PART 2: Write a function to predict the species for each observation

# create a dictionary so we can reference columns by name

# define function that takes in a row of data and returns a predicted species


# make predictions and store as numpy array

# calculate the accuracy of the predictions
