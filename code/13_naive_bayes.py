'''
CLASS: Naive Bayes SMS spam classifier using sklearn
Data source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''

## READING IN THE DATA

# read tab-separated file using pandas
import pandas as pd
df = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/sms.tsv',
                   sep='\t', header=None, names=['label', 'msg'])

# examine the data
df.head(30)
df.label.value_counts()
df.msg.describe()

# convert label to a binary variable
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.msg, df.label, random_state=1)
X_train.shape
X_test.shape


## COUNTVECTORIZER: 'convert text into a matrix of token counts'
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer

# start with a simple example
train_simple = ['Bob Likes Sports',
                'Bob hates sports',
                'Bob really, really likes a beer']

# learn the 'vocabulary' of the training data
vect = CountVectorizer()
vect.fit(train_simple)
vect.get_feature_names()

# transform training data into a 'document-term matrix'
train_simple_dtm = vect.transform(train_simple)
train_simple_dtm
train_simple_dtm.toarray()

# examine the vocabulary and document-term matrix together
pd.DataFrame(train_simple_dtm.toarray(), columns=vect.get_feature_names())

# transform testing data into a document-term matrix (using existing vocabulary)
test_simple = ['Joe really hates beer']
test_simple_dtm = vect.transform(test_simple)
test_simple_dtm.toarray()
pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())


## REPEAT PATTERN WITH SMS DATA

# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm

# store feature names and examine them
train_features = vect.get_feature_names()
len(train_features)
train_features[:50]
train_features[-50:]

# convert train_dtm to a regular array and examine it
train_arr = train_dtm.toarray()
train_arr
sum(train_arr[0, :])
sum(train_arr[:, 0])


## MODEL BUILDING WITH NAIVE BAYES
## http://scikit-learn.org/stable/modules/naive_bayes.html

# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, y_train)

# make predictions on test data using test_dtm
preds = nb.predict(test_dtm)
preds

# compare predictions to true labels
from sklearn import metrics
print metrics.accuracy_score(y_test, preds)
print metrics.confusion_matrix(y_test, preds)

# predict (poorly calibrated) probabilities and calculate AUC
probs = nb.predict_proba(test_dtm)[:, 1]
probs
print metrics.roc_auc_score(y_test, probs)


## COMPARE NAIVE BAYES AND LOGISTIC REGRESSION
## USING ALL DATA AND CROSS-VALIDATION

# create a document-term matrix using all data
all_dtm = vect.fit_transform(df.msg)

# instantiate logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# compare AUC using cross-validation
from sklearn.cross_validation import cross_val_score
cross_val_score(nb, all_dtm, df.label, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, all_dtm, df.label, cv=10, scoring='roc_auc').mean()


## SIMPLE SUMMARIES OF THE DATA

# sum the rows and columns
import numpy as np
tokens_per_email = np.sum(train_arr, axis=1)    # sum of each row
tokens_per_email
count_per_token = np.sum(train_arr, axis=0)     # sum of each column
count_per_token[:50]

# find the most frequent token
np.max(count_per_token)
np.argmax(count_per_token)
train_features[np.argmax(count_per_token)]


## FIND THE 'HAMMIEST' AND 'SPAMMIEST' TOKENS

# split train_arr into ham and spam sections
ham_arr = train_arr[:200]
spam_arr = train_arr[200:]
ham_arr
spam_arr

# calculate count of each token
ham_count_per_token = np.sum(ham_arr, axis=0) + 1
spam_count_per_token = np.sum(spam_arr, axis=0) + 1

# alternative method for accessing counts
ham_count_per_token = nb.feature_count_[0] + 1
spam_count_per_token = nb.feature_count_[1] + 1

# calculate rate of each token
ham_token_rate = ham_count_per_token/float(200)
spam_token_rate = spam_count_per_token/float(200)
ham_token_rate
spam_token_rate

# for each token, calculate ratio of ham-to-spam
ham_to_spam_ratio = ham_token_rate/spam_token_rate
np.max(ham_to_spam_ratio)
ham_arr[:, np.argmax(ham_to_spam_ratio)]        # count of that token in ham emails
spam_arr[:, np.argmax(ham_to_spam_ratio)]       # count of that token in spam emails
train_features[np.argmax(ham_to_spam_ratio)]    # hammiest token

# for each token, calculate ratio of spam-to-ham
spam_to_ham_ratio = spam_token_rate/ham_token_rate
np.max(spam_to_ham_ratio)
spam_arr[:, np.argmax(spam_to_ham_ratio)]       # count of that token in spam emails
ham_arr[:, np.argmax(spam_to_ham_ratio)]        # count of that token in ham emails
train_features[np.argmax(spam_to_ham_ratio)]    # spammiest token
