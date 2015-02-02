'''
HOMEWORK SOLUTION: Glass Identification
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


## PART 1

# read data into a DataFrame
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',
                 header=None, names=['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type'],
                 index_col='id')

# briefly explore the data
df.head()
df.tail()
df.isnull().sum()

# convert to binary classification problem:
#   types 1/2/3/4 are mapped to 0
#   types 5/6/7 are mapped to 1
df['binary'] = np.where(df.glass_type < 5, 0, 1)


## PART 2

# create a list of features (make sure not to use 'id' or 'glass_type' as features!)
feature_cols = ['ri','na','mg','al','si','k','ca','ba','fe']
feature_cols = df.columns[:-2]      # accomplishes the same thing

# define X (features) and y (response)
X = df[feature_cols]
y = df.binary

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)


## PART 3

# fit a logistic regression model and make predictions
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
probs = logreg.predict_proba(X_test)[:, 1]  # predicted probabilities
preds = logreg.predict(X_test)              # predicted classes

# print confusion matrix (must use 'preds')
print metrics.confusion_matrix(y_test, preds)

# calculate accuracy (must use 'preds')
print metrics.accuracy_score(y_test, preds)     # 92.6% accuracy

# calculate null accuracy rate
1 - y_test.mean()                               # 74.1% null accuracy

# calculate AUC (must use 'probs')
print metrics.roc_auc_score(y_test, probs)      # 0.973 AUC


## PART 4

# use cross-validation with AUC as scoring metric to compare models:
#   logistic regression, KNN (K=1), KNN (K=3)
logreg = LogisticRegression()
knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()   # 0.942 AUC
cross_val_score(knn1, X, y, cv=5, scoring='roc_auc').mean()     # 0.861 AUC
cross_val_score(knn3, X, y, cv=5, scoring='roc_auc').mean()     # 0.905 AUC


## PART 5

# group data by 'binary' and see if any features look like good predictors
df.groupby('binary').mean()
df.boxplot(column='mg', by='binary')
df.boxplot(column='al', by='binary')
df.boxplot(column='ba', by='binary')

# try to increase AUC by using a smaller number of features
feature_cols = ['mg','ba']
X = df[feature_cols]
cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()   # 0.989 AUC
