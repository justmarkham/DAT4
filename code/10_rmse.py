'''
CLASS: Model evaluation metrics (RMSE)
'''

## READ DATA AND CREATE DUMMY VARIABLES

# read in the data
import pandas as pd
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# create new feature 'Size', randomly assign as 'small' or 'large'
import numpy as np
np.random.seed(12345)
nums = np.random.rand(len(data))
mask_large = nums > 0.5
data['Size'] = 'small'
data.loc[mask_large, 'Size'] = 'large'

# create dummy variable 'IsLarge'
data['IsLarge'] = data.Size.map({'small':0, 'large':1})

# create new feature 'Area', randomly assign as 'rural' or 'suburban' or 'urban'
np.random.seed(123456)
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'

# create dummy variables 'Area_suburban' and 'Area_urban'
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]
data = pd.concat([data, area_dummies], axis=1)


## CROSS-VALIDATION USING RMSE

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.Sales

# use 10-fold cross-validation to estimate RMSE when including all features
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

# repeat using only the 'meaningful' predictors
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))
