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
http://matplotlib.org/users/mathtext.html
'''
x = np.linspace(-20, 20, 1000)
beta = [0, 0.5]
y = np.exp(beta[0] + beta[1]*x) / (1 + np.exp(beta[0] + beta[1]*x))

# Plotting
plt.plot(x, y, 'r', alpha=0.75, linewidth=2.5)
plt.plot([0,0], [0, 1], 'k')
plt.plot([-20,20], [0.5, 0.5], 'k')
plt.xlabel(r'$x$', fontsize='xx-large')
plt.ylabel(r'$\pi(x)$', fontsize='xx-large')

'''
PART I - Exploration
'''

# 1 - Read in Default.csv and convert all data to numeric
d = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT3/master/data/Default.csv')
d.head()
d.describe()

# Convert everything to numeric before splitting
d.student = np.where(d.student == 'Yes', 1, 0)

# 2 - Split the data into train and test sets
X = d[['balance','student','income']]
y = d.default
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Convert them back into dataframes, for convenience
train = pd.DataFrame(data=X_train, columns=['balance','student','income'])
train['default'] = y_train
test = pd.DataFrame(data=X_test, columns=['balance','student','income'])
test['default'] = y_test
# 3 - Create a histogram of all variables
train.hist()

# 4 - Create a scatter plot of the income vs. balance
train.plot(x='balance', y='income', kind='scatter', alpha=0.3)
plt.ylim([0,80000]); plt.xlim([0, 2800])

# 5 - Mark defaults with a different color and symbol
train_nd = d[d.default == 0]
train_d = d[d.default == 1]

plt.figure()
plt.scatter(train_nd.balance, train_nd.income, alpha = .5, marker='+', c= 'b')

plt.scatter(train_d.balance, train_d.income, marker='o', 
            edgecolors = 'r', facecolors = 'none')
plt.ylim([0,80000]); plt.xlim([0, 2800])
plt.legend( ('default', 'no default'), loc='upper right')

# 6 - What can you infer from this plot?
# It appears that the balance is more correlated with default than income

'''
PART II - LOGISTIC REGRESSION
'''

# 1 - Run a logistic regression on the balance variable
# 2 - Is the beta  value associated with balance significant?
balance = LogisticRegression()
balance.fit(train[['balance']], y_train)
B1 = balance.coef_[0][0]
B0 = balance.intercept_[0]
np.exp(B1)

# Beta is significant!
# 2 - Predict the probability of default for someone with a balance of $1.2k and $2.5k
prob = balance.predict(pd.DataFrame({'balance': [1200, 2500]}))

# What does beta mean? Let's create some plots to find out!
x = np.linspace(test.balance.min(), test.balance.max(),500)
beta = [B0,B1]

y = np.exp(beta[0] + beta[1]*x) / (1 + np.exp(beta[0] + beta[1]*x))
odds = np.exp(beta[0] + beta[1]*x)
log_odds = beta[0] + beta[1]*x

plt.figure(figsize=(7, 8))
plt.subplot(311)
plt.plot(x, y, 'r', linewidth=2)
plt.ylabel('Probability')
plt.text(500, 0.7, r'$\frac{e^{\beta_o + \beta_1x}}{1+e^{\beta_o + \beta_1x}}$', fontsize=25)

plt.subplot(312)
plt.plot(x, odds, 'k', linewidth=2)
plt.ylabel('Odds')
plt.text(500, 10, r'$e^{\beta_o + \beta_1x}$', fontsize=20)

plt.subplot(313)
plt.plot(x, log_odds, 'c', linewidth=2)
plt.ylabel('Log(Odds)')
plt.xlabel('x')
plt.text(500, 1, r'$\beta_o + \beta_1x$', fontsize=15)


'''
From Page 133 in "Introduction to Statistical Learning"
...increasing X by one unit changes the log odds by β1, or equivalently 
it multiplies the odds by e^β1 . However,because the relationship between 
p(X) and X in is not a straight line,β1 does not correspond to the change 
in p(X) associated with a one-unit increase in X. The amount that 
p(X) changes due to a one-unit change in X will depend on the current 
value of X.

Example: In our example, β1 = 0.0042979750671040349 ~= 0.0043

Log-odds: 
If you increase x by 1, you increase the log-odds by 0.0043. 
If you increase x by 800, you increase the log-odds by 0.0043*800 = 3.44

If you increase x by 1, you multiply the odds by e^0.0043. 
If you increase x by 800, you mutliply the odds by e^(0.0043*800) = 31.187, not 800 * e^(0.0043)
'''

# Now let's try plotting some points
plt.subplot(311)
pts = np.array([1200, 2500])
ypts = np.exp(beta[0] + beta[1]*pts) / (1 + np.exp(beta[0] + beta[1]*pts))
plt.plot(pts, ypts, 'ko')

plt.subplot(312)
odds_pts = np.exp(beta[0] + beta[1]*pts)
plt.plot(pts, odds_pts, 'ro')

plt.subplot(313)
log_odds_pts = beta[0] + beta[1]*pts
plt.plot(pts, log_odds_pts , 'ko')



# 3 - Plot the fitted logistic function overtop of the data points
plt.figure()
plt.scatter(test.balance, test.default, alpha=0.1)
plt.plot(x, y, 'r', linewidth=2)
plt.xlabel("Balance")
plt.ylabel("Probability of Default")
plt.ylim([-0.05,1.05]); plt.xlim([0, 2800])
plt.plot([1200, 2500], prob, 'ko')

# 4 - Create predictions using the balance model on the test set
test['pred_class'] = balance.predict(test[['balance']])

# 5 - Compute the overall accuracy, the sensitivity and specificity
# Accuracy
accuracy = sum(test.pred_class == test.default) / float(len(test.default))

# Specificity
# For those who didn't default, how many did it predict correctly?
test_nd = test[test.default == 0]
specificity = sum(test_nd.pred_class == test_nd.default) / float(len(test_nd.default))

# Sensitivity
# For those who did default, how many did it predict correctly? 
test_d = test[test.default == 1]
sensitivity = sum(test_d.pred_class == test_d.default) / float(len(test_d.default))

# This raises the question, how does our overall 
# classification accuracy compare to the not-default rate?
null = 1 - sum(d.default) / float(len(d.default))

# This illustrates an important point, class imbalance can result in accuracy
# measures that are missleading. After all, if you would have just guessed not
# going to default, you would be correct 96.67 % of the time.