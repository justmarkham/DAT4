'''
SOLUTIONS: "Human Learning" with iris data
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

# read iris.data into a pandas DataFrame (df), including column names
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# clean up column names
features = [name[:-5].replace(' ', '_') for name in iris.feature_names]

# read into pandas again, with better column names
df = pd.DataFrame(iris.data, columns=features)

# create a list of species (150 elements) using iris.target and iris.target_names
species = [iris.target_names[num] for num in iris.target]

# add the species list as a new DataFrame column
df['species'] = species

# explore data numerically, looking for differences between species
df.describe()
df.groupby('species').sepal_length.mean()
df.groupby('species')['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].mean()
df.groupby('species').agg(np.mean)
df.groupby('species').agg([np.min, np.max])
df.groupby('species').describe()

# explore data by sorting, looking for differences between species
df.sort_index(by='sepal_length').values
df.sort_index(by='sepal_width').values
df.sort_index(by='petal_length').values
df.sort_index(by='petal_width').values

# explore data visually, looking for differences between species
df.petal_width.hist(by=species, sharex=True)
df.boxplot(column='petal_width', by='species')
df.boxplot(by='species')
df.plot(x='petal_length', y='petal_width', kind='scatter', c=iris.target)
pd.scatter_matrix(df, c=iris.target)


## PART 2: Write a function to predict the species for each observation

# create a dictionary so we can reference columns by name
col_ix = {col:index for index, col in enumerate(df.columns)}

# define function that takes in a row of data and returns a predicted species
def classify_iris(data):
    if data[col_ix['petal_length']] < 3:
        return 'setosa'
    elif data[col_ix['petal_width']] < 1.8:
        return 'versicolor'
    else:
        return 'virginica'

# make predictions and store as numpy array
preds = np.array([classify_iris(row) for row in df.values])

# calculate the accuracy of the predictions
np.mean(preds == df.species.values)
