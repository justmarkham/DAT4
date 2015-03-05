## TASK: Searching for optimal parameters
## FUNCTION: GridSearchCV
## DOCUMENTATION: http://scikit-learn.org/stable/modules/grid_search.html
## DATA: Titanic (n=891, p=5 selected, type=classification)
## DATA DICTIONARY: https://www.kaggle.com/c/titanic-gettingStarted/data

# read in and prepare titanic data
import pandas as pd
titanic = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/titanic.csv')
titanic['sex'] = titanic.sex.map({'female':0, 'male':1})
titanic.age.fillna(titanic.age.mean(), inplace=True)
embarked_dummies = pd.get_dummies(titanic.embarked, prefix='embarked').iloc[:, 1:]
titanic = pd.concat([titanic, embarked_dummies], axis=1)

# define X and y
feature_cols = ['pclass', 'sex', 'age', 'embarked_Q', 'embarked_S']
X = titanic[feature_cols]
y = titanic.survived

# use cross-validation to find best max_depth
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

# try max_depth=2
treeclf = DecisionTreeClassifier(max_depth=2, random_state=1)
cross_val_score(treeclf, X, y, cv=10, scoring='roc_auc').mean()

# try max_depth=3
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
cross_val_score(treeclf, X, y, cv=10, scoring='roc_auc').mean()

# use GridSearchCV to automate the search
from sklearn.grid_search import GridSearchCV
treeclf = DecisionTreeClassifier(random_state=1)
max_depth_range = range(1, 21)
param_grid = dict(max_depth=max_depth_range)
grid = GridSearchCV(treeclf, param_grid, cv=10, scoring='roc_auc')
grid.fit(X, y)

# check the results of the grid search
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
grid_mean_scores

# plot the results
import matplotlib.pyplot as plt
plt.plot(max_depth_range, grid_mean_scores)

# what was best?
grid.best_score_
grid.best_params_
grid.best_estimator_

# search a "grid" of parameters
max_depth_range = range(1, 21)
min_samples_leaf_range = range(1, 11)
param_grid = dict(max_depth=max_depth_range, min_samples_leaf=min_samples_leaf_range)
grid = GridSearchCV(treeclf, param_grid, cv=10, scoring='roc_auc')
grid.fit(X, y)
grid.best_score_
grid.best_params_


## TASK: Standardization of features (aka "center and scale" or "z-score normalization")
## FUNCTION: StandardScaler
## DOCUMENTATION: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
## EXAMPLE: http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/about_standardization_normalization.ipynb
## DATA: Wine (n=178, p=2 selected, type=classification)
## DATA DICTIONARY: http://archive.ics.uci.edu/ml/datasets/Wine

# sample data
train = pd.DataFrame({'A':[40,50,60], 'B':[0.90,0.30,0.60], 'C':[0,0.20,0.80], 'label':[0,1,2]})
oos = pd.DataFrame({'A':[54.9], 'B':[0.59], 'C':[0.79]})

# define X and y
X = train[['A','B','C']]
y = train.label

# KNN with k=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# what "should" it predict? what does it predict?
knn.predict(oos)

# standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# compare original to standardized
X.values
X_scaled

# figure out how it standardized
scaler.mean_
scaler.std_
(X.values-scaler.mean_) / scaler.std_

# try this on real data
wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None, usecols=[0,10,13])
wine.columns=['label', 'color', 'proline']
wine.head()
wine.describe()

# define X and y
X = wine[['color', 'proline']]
y = wine.label

# split into train/test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# standardize
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# check that it worked properly
X_train_scaled[:, 0].mean()
X_train_scaled[:, 0].std()
X_train_scaled[:, 1].mean()
X_train_scaled[:, 1].std()

# standardize X_test
X_test_scaled = scaler.transform(X_test)

# is this right?
X_test_scaled[:, 0].mean()
X_test_scaled[:, 0].std()
X_test_scaled[:, 1].mean()
X_test_scaled[:, 1].std()

# compare KNN accuracy on original vs scaled data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)


## TASK: Chaining steps
## FUNCTION: Pipeline
## DOCUMENTATION: http://scikit-learn.org/stable/modules/pipeline.html
## DATA: Wine (n=178, p=2 selected, type=classification)
## DATA DICTIONARY: http://archive.ics.uci.edu/ml/datasets/Wine

# here is proper cross-validation on the original (unscaled) data
X = wine[['color', 'proline']]
y = wine.label
knn = KNeighborsClassifier(n_neighbors=3)
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()

# why is this improper cross-validation on the scaled data?
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy').mean()

# fix this using Pipeline
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

# using GridSearchCV with Pipeline
n_neighbors_range = range(1, 21)
param_grid = dict(kneighborsclassifier__n_neighbors=n_neighbors_range)
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
grid.best_score_
grid.best_params_


## TASK: Regularized regression
## FUNCTIONS: Ridge, RidgeCV, Lasso, LassoCV
## DOCUMENTATION: http://scikit-learn.org/stable/modules/linear_model.html
## DATA: Crime (n=319 non-null, p=122, type=regression)
## DATA DICTIONARY: http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

# read in data, remove categorical features, remove rows with missing values
crime = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data', header=None, na_values=['?'])
crime = crime.iloc[:, 5:]
crime.dropna(inplace=True)

# define X and y
X = crime.iloc[:, :-1]
y = crime.iloc[:, -1]

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_

# make predictions and evaluate
import numpy as np
from sklearn import metrics
preds = lm.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))

# ridge regression (alpha must be positive, larger means more regularization)
from sklearn.linear_model import Ridge
rreg = Ridge(alpha=0.1, normalize=True)
rreg.fit(X_train, y_train)
rreg.coef_
preds = rreg.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))

# use RidgeCV to select best alpha
from sklearn.linear_model import RidgeCV
alpha_range = 10.**np.arange(-2, 3)
rregcv = RidgeCV(normalize=True, scoring='mean_squared_error', alphas=alpha_range)
rregcv.fit(X_train, y_train)
rregcv.alpha_
preds = rregcv.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))

# lasso (alpha must be positive, larger means more regularization)
from sklearn.linear_model import Lasso
las = Lasso(alpha=0.01, normalize=True)
las.fit(X_train, y_train)
las.coef_
preds = las.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))

# try a smaller alpha
las = Lasso(alpha=0.0001, normalize=True)
las.fit(X_train, y_train)
las.coef_
preds = las.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))

# use LassoCV to select best alpha (tries 100 alphas by default)
from sklearn.linear_model import LassoCV
lascv = LassoCV(normalize=True)
lascv.fit(X_train, y_train)
lascv.alpha_
lascv.coef_
preds = lascv.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))


## TASK: Regularized classification
## FUNCTION: LogisticRegression
## DOCUMENTATION: http://scikit-learn.org/stable/modules/linear_model.html
## DATA: Titanic (n=891, p=5 selected, type=classification)
## DATA DICTIONARY: https://www.kaggle.com/c/titanic-gettingStarted/data

# define X and y
feature_cols = ['pclass', 'sex', 'age', 'embarked_Q', 'embarked_S']
X = titanic[feature_cols]
y = titanic.survived

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.coef_

# logistic regression with L2 penalty (C must be positive, smaller means more regularization)
logreg = LogisticRegression(C=0.5, penalty='l2')
logreg.fit(X_train, y_train)
logreg.coef_

# pipeline with scaling to select best C and penalty
pipe = make_pipeline(StandardScaler(), LogisticRegression())
C_range = 10.**np.arange(-2, 3)
penalty_options = ['l1', 'l2']
param_grid = dict(logisticregression__C=C_range, logisticregression__penalty=penalty_options)
grid = GridSearchCV(pipe, param_grid, cv=10, scoring='roc_auc')
grid.fit(X, y)
grid.best_score_
grid.best_params_


## TASK: Feature selection
## FUNCTIONS: RFE, RFECV
## DOCUMENTATION: http://scikit-learn.org/stable/modules/feature_selection.html
## DATA: Crime (n=319 non-null, p=122, type=regression)
## DATA DICTIONARY: http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

# define X and y
X = crime.iloc[:, :-1]
y = crime.iloc[:, -1]

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# select "best" features (half of them by default)
lm = LinearRegression()
from sklearn.feature_selection import RFE
selector = RFE(lm)
selector.fit(X_train, y_train)
selector.n_features_
selector.support_
selector.ranking_

# let RFECV select the "optimal" number of features
from sklearn.feature_selection import RFECV
selector = RFECV(lm, cv=3, scoring='mean_squared_error')
selector.fit(X, y)
selector.n_features_
selector.support_
selector.ranking_

# *tentative* advice for usage:
# 1. scale features, then use RFECV to select the number of features (p)
# 2. build pipeline: feature scaling, select p features using RFE, model
# 3. GridSearchCV to select optimal parameters
