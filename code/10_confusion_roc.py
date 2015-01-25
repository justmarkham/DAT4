'''
CLASS: Model evaluation metrics (confusion matrix, ROC/AUC)
'''

## READ DATA AND SPLIT INTO TRAIN/TEST

# read in the data
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT3/master/data/Default.csv')

# create X and y
X = data[['balance']]
y = data.default

# split into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


## CALCULATE ACCURACY

# predict using logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)

# calculate accuracy
from sklearn import metrics
print metrics.accuracy_score(y_test, preds)

# predict and calculate accuracy in one step
logreg.score(X_test, y_test)

# compare to null accuracy rate
1 - y_test.mean()


## CONFUSION MATRIX

# print confusion matrix
print metrics.confusion_matrix(y_test, preds)

# nicer confusion matrix
from nltk import ConfusionMatrix
print ConfusionMatrix(list(y_test), list(preds))

# sensitivity: percent of correct predictions when reference value is 'default'
21 / float(58 + 21)
print metrics.recall_score(y_test, preds)

# specificity: percent of correct predictions when reference value is 'not default'
2416 / float(2416 + 5)

# change cutoff for predicting default to 0.2
import numpy as np
probs = logreg.predict_proba(X_test)[:, 1]
preds = np.where(probs > 0.2, 1, 0)

# check accuracy, sensitivity, specificity
print metrics.accuracy_score(y_test, preds)
print ConfusionMatrix(list(y_test), list(preds))
45 / float(34 + 45)
2340 / float(2340 + 81)


## ROC CURVES and AUC

# plot ROC curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# calculate AUC
print metrics.roc_auc_score(y_test, probs)

# use AUC as evaluation metric for cross-validation
from sklearn.cross_validation import cross_val_score
X = data[['balance']]
y = data.default
logreg = LogisticRegression()
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

# compare to a model with an additional feature
X = data[['balance', 'income']]
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
