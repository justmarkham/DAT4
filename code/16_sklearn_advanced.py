## SEARCHING FOR OPTIMAL PARAMETERS: GridSearchCV
## http://scikit-learn.org/stable/modules/grid_search.html

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
treeclf = DecisionTreeClassifier()
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


## STANDARDIZATION OF FEATURES: StandardScaler
## AKA "CENTER AND SCALE" or "Z-SCORE NORMALIZATION"
## http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
## http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/about_standardization_normalization.ipynb

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


## CHAINING STEPS: Pipeline
## http://scikit-learn.org/stable/modules/pipeline.html

# here is proper cross-validation on the original (unscaled) data
X = wine[['color', 'proline']]
y = wine.label
knn = KNeighborsClassifier(n_neighbors=3)
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()

# why is this improper cross-validation on the scaled date?
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
