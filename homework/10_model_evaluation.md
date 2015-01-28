## Class 10 Homework: Model Evaluation

Practice what we've learned in class using the [Glass Identification Data Set](http://archive.ics.uci.edu/ml/datasets/Glass+Identification). This is due by midnight on Sunday.

* Part 1:
    * Read the data into a DataFrame.
    * Briefly explore the data to make sure the DataFrame matches your expectations.
    * Create a new column called 'binary' that maps the glass type from 7 classes to 2 classes:
        * If type of glass = 1/2/3/4, binary = 0.
        * If type of glass = 5/6/7, binary = 1.
* Part 2:
    * Create 'X' using all features, and create 'y' using the binary column.
        * Don't use the ID number or the glass type as features!
    * Split X and y into training and test sets.
* Part 3:
    * Fit a logistic regression model on your training set, and make predictions on your test set.
    * Print the confusion matrix.
    * Calculate the accuracy and compare it to the null accuracy rate.
    * Calculate the AUC.
* Part 4:
    * Use cross-validation (with AUC as the scoring metric) to compare these three models:
        * logistic regression
        * KNN (K = 1)
        * KNN (K = 3)
* Part 5 (Optional):
    * Explore the data to see if any features look like good predictors.
    * Use cross-validation to compare a model with a smaller set of features with your best model from Part 4.
