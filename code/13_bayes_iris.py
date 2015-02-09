'''
CLASS: Applying Bayes' theorem to iris classification
'''

# load the iris data
from sklearn.datasets import load_iris
iris = load_iris()

# round up the measurements
import numpy as np
X = np.ceil(iris.data)

# clean up column names
features = [name[:-5].replace(' ', '_') for name in iris.feature_names]

# read into pandas
import pandas as pd
df = pd.DataFrame(X, columns=features)

# create a list of species using iris.target and iris.target_names
species = [iris.target_names[num] for num in iris.target]

# add the species list as a new DataFrame column
df['species'] = species

# print the DataFrame
df

# show all observations with features: 7, 3, 5, 2
df[(df.sepal_length==7) & (df.sepal_width==3) & (df.petal_length==5) & (df.petal_width==2)]
