'''
L O G I S T I C   R E G R E S S I O N
Adapted From example given in Chapter 4 of 
Introduction to Statistical Learning
Data: Default Data Set
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

'''
QUIZ: UNDERSTANDING THE BASIC SHAPE
'''


'''
PART I - Exploration
'''

# 1 - Read in Default.csv and convert all data to numeric

# Convert everything to numeric before splitting

# 2 - Split the data into train and test sets

# Can convert arrays back into dataframes if desired, for convenience later on

# 3 - Create a histogram of all variables

# 4 - Create a scatter plot of the income vs. balance

# 5 - Mark defaults with a different color and symbol

# 6 - What can you infer from this plot?



'''
PART II - LOGISTIC REGRESSION
'''

# 1 - Run a logistic regression on the balance variable

# 2 - Is the beta value associated with balance significant?

# 3 - Predict the probability of default for someone with a balance of $1.2k and $2.5k

# 4 - Plot the fitted logistic function overtop of the data points

# 5 - Create predictions using the test set

# 6 - Compute the overall accuracy, the sensitivity and specificity
# Accuracy
# How many were classified correctly?

# Specificity
# For those who didn't default, how many did it predict correctly?

# Sensitivity
# For those who did default, how many did it predict correctly?