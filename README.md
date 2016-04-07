## DAT4 Course Repository

Course materials for [General Assembly's Data Science course](https://generalassemb.ly/education/data-science/washington-dc/) in Washington, DC (12/15/14 - 3/16/15).

**Instructors:** Sinan Ozdemir and Kevin Markham ([Data School blog](http://www.dataschool.io/), [email newsletter](http://www.dataschool.io/subscribe/), [YouTube channel](https://www.youtube.com/user/dataschool))

**Teaching Assistant:** Brandon Burroughs

**Office hours:** 1-3pm on Saturday and Sunday ([Starbucks at 15th & K](http://www.yelp.com/biz/starbucks-washington-15)), 5:15-6:30pm on Monday (GA)

**[Course Project information](project.md)**

Monday | Wednesday
--- | ---
12/15: [Introduction](#class-1-introduction) | 12/17: [Python](#class-2-python)
12/22: [Getting Data](#class-3-getting-data) | 12/24: *No Class*
12/29: *No Class* | 12/31: *No Class*
1/5: [Git and GitHub](#class-4-git-and-github) | 1/7: [Pandas](#class-5-pandas)<br>**Milestone:** Question and Data Set
1/12: [Numpy, Machine Learning, KNN](#class-6-numpy-machine-learning-knn) | 1/14: [scikit-learn, Model Evaluation Procedures](#class-7-scikit-learn-model-evaluation-procedures)
1/19: *No Class* | 1/21: [Linear Regression](#class-8-linear-regression)
1/26: [Logistic Regression,<br>Preview of Other Models](#class-9-logistic-regression-preview-of-other-models) | 1/28: [Model Evaluation Metrics](#class-10-model-evaluation-metrics)<br>**Milestone:** Data Exploration and Analysis Plan
2/2: [Working a Data Problem](#class-11-working-a-data-problem) | 2/4: [Clustering and Visualization](#class-12-clustering-and-visualization)<br>**Milestone:** Deadline for Topic Changes
2/9: [Naive Bayes](#class-13-naive-bayes) | 2/11: [Natural Language Processing](#class-14-natural-language-processing)
2/16: *No Class* | 2/18: [Decision Trees](#class-15-decision-trees)<br>**Milestone:** First Draft
2/23: [Ensembling](#class-16-ensembling) | 2/25: [Databases and MapReduce](#class-17-databases-and-mapreduce)
3/2: [Recommenders](#class-18-recommenders) | 3/4: [Advanced scikit-learn](#class-19-advanced-scikit-learn)<br>**Milestone:** Second Draft (Optional)
3/9: [Course Review](#class-20-course-review) | 3/11: [Project Presentations](#class-21-project-presentations)
3/16: [Project Presentations](#class-22-project-presentations) |


### Installation and Setup
* Install the [Anaconda distribution](http://continuum.io/downloads) of Python 2.7x.
* Install [Git](http://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and create a [GitHub](https://github.com/) account.
* Once you receive an email invitation from [Slack](https://slack.com/), join our "DAT4 team" and add your photo!


### Class 1: Introduction
* Introduction to General Assembly
* Course overview: our philosophy and expectations ([slides](slides/01_course_overview.pdf))
* Data science overview ([slides](slides/01_intro_to_data_science.pdf))
* Tools: check for proper setup of Anaconda, overview of Slack

**Homework:**
* Resolve any installation issues before next class.

**Optional:**
* Review the [code](code/00_python_refresher.py) from Saturday's Python refresher for a recap of some Python basics.
* Read [Analyzing the Analyzers](http://cdn.oreillystatic.com/oreilly/radarreport/0636920029014/Analyzing_the_Analyzers.pdf) for a useful look at the different types of data scientists.
* Subscribe to the [Data Community DC newsletter](http://www.datacommunitydc.org/thenewsletter/) or check out their [event calendar](http://www.datacommunitydc.org/calendar) to become acquainted with the local data community.


### Class 2: Python
* Brief overview of Python environments: Python interpreter, IPython interpreter, Spyder
* Python quiz ([solution](code/02_python_quiz_solution.py))
* Working with data in Python
    * Obtain data from a [public data source](public_data.md)
    * [FiveThirtyEight alcohol data](https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption), and [revised data](data/drinks.csv) (continent column added)
    * Reading and writing files in Python ([code](code/02_file_io.py))

**Homework:**
* [Python exercise](code/02_file_io_homework.py) ([solution](code/02_file_io_homework_solution.py))
* Read through the [project page](project.md) in detail.
* Review a few [projects from past Data Science courses](https://github.com/justmarkham/DAT-project-examples) to get a sense of the variety and scope of student projects.
    * Check for proper setup of Git by running `git clone https://github.com/justmarkham/DAT-project-examples.git`

**Optional:**
* If you need more practice with Python, review the "Python Overview" section of [A Crash Course in Python](http://nbviewer.ipython.org/gist/rpmuller/5920182), work through some of [Codecademy's Python course](http://www.codecademy.com/en/tracks/python), or work through [Google's Python Class](https://developers.google.com/edu/python/) and its exercises.
* For more project inspiration, browse the [student projects](http://cs229.stanford.edu/projects2013.html) from Andrew Ng's [Machine Learning course](http://cs229.stanford.edu/) at Stanford.

**Resources:**
* [Online Python Tutor](http://pythontutor.com/) is useful for visualizing (and debugging) your code.


### Class 3: Getting Data
* Checking your homework
* Regular expressions, web scraping, APIs ([slides](slides/03_getting_data.pdf), [regex code](code/03_re_example.py), [web scraping and API code](code/03_getting_data.py))
* Any questions about the course project?

**Homework:**
* Think about your project question, and start looking for data that will help you to answer your question.
* Prepare for our next class on Git and GitHub:
    * You'll need to know some command line basics, so please work through GA's excellent [command line tutorial](http://generalassembly.github.io/prework/command-line/#/) and then take this brief [quiz](https://gahub.typeform.com/to/J6xirf).
    * Check for proper setup of Git by running `git clone https://github.com/justmarkham/DAT-project-examples.git`. If that doesn't work, you probably need to [install Git](http://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
    * Create a [GitHub account](https://github.com/). (You don't need to download anything from GitHub.)

**Optional:**
* If you aren't feeling comfortable with the Python we've done so far, keep practicing using the resources above!

**Resources:**
* [regex101](https://regex101.com/#python) is an excellent tool for testing your regular expressions. For learning more regular expressions, Google's Python Class includes an [excellent regex lesson](https://developers.google.com/edu/python/regular-expressions) (which includes a [video](http://www.youtube.com/watch?v=kWyoYtvJpe4)).
* [Mashape](https://www.mashape.com/explore) and [Apigee](https://apigee.com/providers) allow you to explore tons of different APIs. Alternatively, a [Python API wrapper](http://www.pythonforbeginners.com/api/list-of-python-apis) is available for many popular APIs.


### Class 4: Git and GitHub
* Special guest: Nick DePrey presenting his class project from DAT2
* Git and GitHub ([slides](slides/04_git_github.pdf))

**Homework:**
* Project milestone: Submit your [question and data set](project.md) to your folder in [DAT4-students](https://github.com/justmarkham/DAT4-students) before class on Wednesday! (This is a great opportunity to practice writing Markdown and creating a pull request.)

**Optional:**
* Clone this repo (DAT4) for easy access to the course files.

**Resources:**
* Read the first two chapters of [Pro Git](http://git-scm.com/book/en/v2) to gain a much deeper understanding of version control and basic Git commands.
* [GitRef](http://gitref.org/) is an excellent reference guide for Git commands.
* [Git quick reference for beginners](http://www.dataschool.io/git-quick-reference-for-beginners/) is a shorter reference guide with commands grouped by workflow.
* The [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) covers standard Markdown and a bit of "[GitHub Flavored Markdown](https://help.github.com/articles/github-flavored-markdown/)."


### Class 5: Pandas
* Pandas for data exploration, analysis, and visualization ([code](code/05_pandas.py))
    * [Split-Apply-Combine](http://i.imgur.com/yjNkiwL.png) pattern
    * Simple examples of [joins in Pandas](http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/#joining)

**Homework:**
* [Pandas homework](homework/05_pandas.md)

**Optional:**
* To learn more Pandas, review this [three-part tutorial](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/), or review these three excellent (but extremely long) notebooks on Pandas: [introduction](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_5-Introduction-to-Pandas.ipynb), [data wrangling](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_6-Data-Wrangling-with-Pandas.ipynb), and [plotting](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_7-Plotting-with-Pandas.ipynb).

**Resources:**
* For more on Pandas plotting, read the [visualization page](http://pandas.pydata.org/pandas-docs/stable/visualization.html) from the official Pandas documentation.
* To learn how to customize your plots further, browse through this [notebook on matplotlib](http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section2_4-Matplotlib.ipynb).
* To explore different types of visualizations and when to use them, [Choosing a Good Chart](http://www.extremepresentation.com/uploads/documents/choosing_a_good_chart.pdf) is a handy one-page reference, and Columbia's Data Mining class has an excellent [slide deck](http://www2.research.att.com/~volinsky/DataMining/Columbia2011/Slides/Topic2-EDAViz.ppt).


### Class 6: Numpy, Machine Learning, KNN
* Numpy ([code](code/06_numpy.py))
* "Human learning" with iris data ([code](code/06_iris_prework.py), [solution](code/06_iris_solution.py))
* Machine Learning and K-Nearest Neighbors ([slides](slides/06_ml_knn.pdf))

**Homework:**
* Read this excellent article, [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html), and be prepared to discuss it in class on Wednesday. (You can ignore sections 4.2 and 4.3.) Here are some questions to think about while you read:
    * In the Party Registration example, what are the features? What is the response? Is this a regression or classification problem?
    * In the interactive visualization, try using different values for K across different sets of training data. What value of K do you think is "best"? How do you define "best"?
    * In the visualization, what do the lighter colors versus the darker colors mean? How is the darkness calculated?
    * How does the choice of K affect model bias? How about variance?
    * As you experiment with K and generate new training data, how can you "see" high versus low variance? How can you "see" high versus low bias?
    * Why should we care about variance at all? Shouldn't we just minimize bias and ignore variance?
    * Does a high value for K cause over-fitting or under-fitting?

**Resources:**
* For a more in-depth look at machine learning, read section 2.1 (14 pages) of Hastie and Tibshirani's excellent book, [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). (It's a free PDF download!)


### Class 7: scikit-learn, Model Evaluation Procedures
* Introduction to scikit-learn with iris data ([code](code/07_sklearn_knn.py))
* Exploring the scikit-learn documentation: [user guide](http://scikit-learn.org/stable/modules/neighbors.html), [module reference](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors), [class documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* Discuss the [article](http://scott.fortmann-roe.com/docs/BiasVariance.html) on the bias-variance tradeoff
* Model evaluation procedures ([slides](slides/07_model_evaluation_procedures.pdf), [code](code/07_model_evaluation_procedures.py))

**Homework:**
* Keep working on your project. Your [data exploration and analysis plan](project.md) is due in two weeks!

**Optional:**
* Practice what we learned in class today!
    * If you have gathered your project data already: Try using KNN for classification, and then evaluate your model. Don't worry about using all of your features, just focus on getting the end-to-end process working in scikit-learn. (Even if your project is regression instead of classification, you can easily convert a regression problem into a classification problem by converting numerical ranges into categories.)
    * If you don't yet have your project data: Pick a suitable dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html), try using KNN for classification, and evaluate your model. The [Glass Identification Data Set](http://archive.ics.uci.edu/ml/datasets/Glass+Identification) is a good one to start with.
    * Either way, you can submit your commented code to DAT4-students, and we'll give you feedback.

**Resources:**
* Here's a great [30-second explanation of overfitting](http://www.quora.com/What-is-an-intuitive-explanation-of-overfitting/answer/Jessica-Su).
* For more on today's topics, these videos from Hastie and Tibshirani are useful: [overfitting and train/test split](https://www.youtube.com/watch?v=_2ij6eaaSl0) (14 minutes), [cross-validation](https://www.youtube.com/watch?v=nZAM5OXrktY) (14 minutes). (Note that they use the terminology "validation set" instead of "test set".)
    * Alternatively, read section 5.1 (12 pages) of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), which covers the same content as the videos.
* This video from Caltech's machine learning course presents an [excellent, simple example of the bias-variance tradeoff](http://work.caltech.edu/library/081.html) (15 minutes) that may help you to visualize bias and variance.


### Class 8: Linear Regression
* Linear regression ([IPython notebook](http://nbviewer.ipython.org/github/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb))

**Homework:**
* Keep working on your project. Your [data exploration and analysis plan](project.md) is due next Wednesday!

**Optional:**
* Similar to last class, your optional exercise is to practice what we have been learning in class, either on your project data or on another dataset.

**Resources:**
* To go much more in-depth on linear regression, read Chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), from which this lesson was adapted. Alternatively, watch the [related videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) or read my [quick reference guide](http://www.dataschool.io/applying-and-interpreting-linear-regression/) to the key points in that chapter.
* To learn more about Statsmodels and how to interpret the output, DataRobot has some decent posts on [simple linear regression](http://www.datarobot.com/blog/ordinary-least-squares-in-python/) and [multiple linear regression](http://www.datarobot.com/blog/multiple-regression-using-statsmodels/).
* This [introduction to linear regression](http://people.duke.edu/~rnau/regintro.htm) is much more detailed and mathematically thorough, and includes lots of good advice.
* This is a relatively quick post on the [assumptions of linear regression](http://pareonline.net/getvn.asp?n=2&v=8).


### Class 9: Logistic Regression, Preview of Other Models
* Logistic regression ([slides](slides/09_logistic_regression.pdf), [exercise](code/09_logistic_regression_exercise.py), [solution](code/09_logistic_regression_class.py))
* Preview of other models

**Resources:**
* For more on logistic regression, watch the [first three videos](https://www.youtube.com/playlist?list=PL5-da3qGB5IC4vaDba5ClatUmFppXLAhE) (30 minutes total) from Chapter 4 of An Introduction to Statistical Learning.
* UCLA's IDRE has a handy table to help you remember the [relationship between probability, odds, and log-odds](http://www.ats.ucla.edu/stat/mult_pkg/faq/general/odds_ratio.htm).
* Better Explained has a very friendly introduction (with lots of examples) to the [intuition behind "e"](http://betterexplained.com/articles/an-intuitive-guide-to-exponential-functions-e/).
* Here are some useful lecture notes on [interpreting logistic regression coefficients](http://www.unm.edu/~schrader/biostat/bio2/Spr06/lec11.pdf).


### Class 10: Model Evaluation Metrics
* Finishing model evaluation procedures ([slides](slides/07_model_evaluation_procedures.pdf), [code](code/07_model_evaluation_procedures.py))
    * Review of test set approach
    * Cross-validation
* Model evaluation metrics ([slides](slides/10_model_evaluation_metrics.pdf))
    * Regression:
        * Root Mean Squared Error ([code](code/10_rmse.py))
    * Classification:
        * Confusion matrix ([code](code/10_confusion_roc.py))
        * ROC curve ([video](https://www.youtube.com/watch?v=OAl6eAyP-yo))

**Homework:**
* [Model evaluation homework](homework/10_model_evaluation.md), due by midnight on Sunday.
    * [Sample solution code](code/10_glass_id_homework_solution.py).
* Watch Kevin's [Kaggle project presentation video](https://www.youtube.com/watch?v=HGr1yQV3Um0) (16 minutes) for an overview of the end-to-end machine learning process, including some aspects that we have not yet covered in class.
* Read this short article on Google's [Smart Autofill](http://googleresearch.blogspot.com/2014/10/smart-autofill-harnessing-predictive.html), and see if you can figure out exactly how the system works.

**Optional:**
* For more on Kaggle, watch [Kaggle Transforms Data Science Into Competitive Sport](https://www.youtube.com/watch?v=8w4UY66GKcM) (28 minutes).

**Resources:**
* scikit-learn has extensive documentation on [model evaluation](http://scikit-learn.org/stable/modules/model_evaluation.html).
* The Kaggle wiki has a decent page describing other common [model evaluation metrics](https://www.kaggle.com/wiki/Metrics).
* Kevin wrote a [simple guide to confusion matrix terminology](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) that you can use as a reference guide.
* Kevin's [blog post about the ROC video](http://www.dataschool.io/roc-curves-and-auc-explained/) includes the complete transcript and screenshots, in case you learn better by reading instead of watching.
* Rahul Patwari has two excellent and highly accessible videos on [Sensitivity and Specificity](https://www.youtube.com/watch?v=U4_3fditnWg&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (9 minutes) and [ROC Curves](https://www.youtube.com/watch?v=21Igj5Pr6u4&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (12 minutes).


### Class 11: Working a Data Problem
* Today we will work on a real world data problem! Our [data](data/ZYX_prices.csv) is stock data over 7 months of a fictional company ZYX including twitter sentiment, volume and stock price. Our goal is to create a predictive model that predicts forward returns.

* Project overview ([slides](slides/11_GA_Stocks.pdf))
    * Be sure to read documentation thoroughly and ask questions! We may not have included all of the information you need...


### Class 12: Clustering and Visualization
* The [slides](slides/12_clustering.pdf) today will focus on our first look at unsupervised learning, K-Means Clustering!
* The [code](code/) for today focuses on two main examples:
    * We will investigate simple clustering using the iris data set.
    * We will take a look at a harder example, using Pandora songs as data. See [data](data/songs.csv).

**Homework:**
* Read Paul Graham's [A Plan for Spam](http://www.paulgraham.com/spam.html) and be prepared to **discuss it in class on Monday**. Here are some questions to think about while you read:
    * Should a spam filter optimize for sensitivity or specificity, in Paul's opinion?
    * Before he tried the "statistical approach" to spam filtering, what was his approach?
    * How exactly does his statistical filtering system work?
    * What did Paul say were some of the benefits of the statistical approach?
    * How good was his prediction of the "spam of the future"?
* Below are the foundational topics upon which Monday's class will depend. Please review these materials before class:
    * **Confusion matrix:** [Kevin's guide](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) roughly mirrors the lecture from class 10.
    * **Sensitivity and specificity:** Rahul Patwari has an [excellent video](https://www.youtube.com/watch?v=U4_3fditnWg&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (9 minutes).
    * **Basics of probability:** These [introductory slides](https://docs.google.com/presentation/d/1cM2dVbJgTWMkHoVNmYlB9df6P2H8BrjaqAcZTaLe9dA/edit#slide=id.gfc3caad2_00) (from the [OpenIntro Statistics textbook](https://www.openintro.org/stat/textbook.php)) are quite good and include integrated quizzes. Pay specific attention to these terms: probability, sample space, mutually exclusive, independent.
* You should definitely be working on your project! **Your rough draft is due in two weeks!**

**Resources:**
* [Introduction to Data Mining](http://www-users.cs.umn.edu/~kumar/dmbook/index.php) has a nice [chapter on cluster analysis](http://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf).
* The scikit-learn user guide has a nice [section on clustering](http://scikit-learn.org/stable/modules/clustering.html).


### Class 13: Naive Bayes
* Briefly discuss [A Plan for Spam](http://www.paulgraham.com/spam.html)
* Probability and Bayes' theorem
    * [Slides](slides/13_naive_bayes.pdf) part 1
    * [Visualization of conditional probability](http://setosa.io/conditional/)
    * Applying Bayes' theorem to iris classification ([code](code/13_bayes_iris.py))
* Naive Bayes classification
    * [Slides](slides/13_naive_bayes.pdf) part 2
    * Example with spam email
    * [Airport security example](http://www.quora.com/In-laymans-terms-how-does-Naive-Bayes-work/answer/Konstantin-Tt)
* Naive Bayes classification in scikit-learn ([code](code/13_naive_bayes.py))
    * Data set: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
    * scikit-learn documentation: [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)

**Resources:**
* The first part of the slides was adapted from [Visualizing Bayes' theorem](http://oscarbonilla.com/2009/05/visualizing-bayes-theorem/), which includes an additional example (using Venn diagrams) of how this applies to testing for breast cancer.
* For an alternative introduction to Bayes' Theorem, [Bayes' Rule for Ducks](https://planspacedotorg.wordpress.com/2014/02/23/bayes-rule-for-ducks/), this [5-minute video on conditional probability](https://www.youtube.com/watch?v=Zxm4Xxvzohk), or these [slides on conditional probability](https://docs.google.com/presentation/d/1psUIyig6OxHQngGEHr3TMkCvhdLInnKnclQoNUr4G4U/edit#slide=id.gfc69f484_00) may be helpful.
* For more details on Naive Bayes classification, Wikipedia has two useful articles ([Naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) and [Naive Bayes spam filtering](http://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)), and Cross Validated has an excellent [Q&A](http://stats.stackexchange.com/questions/21822/understanding-naive-bayes).
* If you enjoyed Paul Graham's article, you can read [his follow-up article](http://www.paulgraham.com/better.html) on how he improved his spam filter and this [related paper](http://www.merl.com/publications/docs/TR2004-091.pdf) about state-of-the-art spam filtering in 2004.

**Homework:**
* Download all of the NLTK collections.
   * In Python, use the following commands to bring up the download menu.
   * ```import nltk```
   * ```nltk.download()```
   * Choose "all".
   * Alternatively, just type ```nltk.download('all')```
* Install two new packages:  ```textblob``` and ```lda```.
   * Open a terminal or command prompt.
   * Type ```pip install textblob``` and ```pip install lda```.


### Class 14: Natural Language Processing
* Overview of Natural Language Processing ([slides](slides/14_natural_language_processing.pdf))
* Real World Examples
* Natural Language Processing ([code](code/14_nlp_class.py))
* NLTK: tokenization, stemming, lemmatization, part of speech tagging, stopwords, Named Entity Recognition (Stanford NER Tagger), TF-IDF, LDA, document summarization
* Alternative: TextBlob

**Resources:**
* [Natural Language Processing with Python](http://www.nltk.org/book/): free online book to go in-depth with NLTK
* [NLP online course](https://www.coursera.org/course/nlp): no sessions are available, but [video lectures](https://class.coursera.org/nlp/lecture) and [slides](http://web.stanford.edu/~jurafsky/NLPCourseraSlides.html) are still accessible
* [Brief slides](http://files.meetup.com/7616132/DC-NLP-2013-09%20Charlie%20Greenbacker.pdf) on the major task areas of NLP
* [Detailed slides](https://github.com/ga-students/DAT_SF_9/blob/master/16_Text_Mining/DAT9_lec16_Text_Mining.pdf) on a lot of NLP terminology
* [A visual survey of text visualization techniques](http://textvis.lnu.se/): for exploration and inspiration
* [DC Natural Language Processing](http://www.meetup.com/DC-NLP/): active Meetup group
* [Stanford CoreNLP](http://nlp.stanford.edu/software/corenlp.shtml): suite of tools if you want to get serious about NLP
* Getting started with regex: [Python introductory lesson](https://developers.google.com/edu/python/regular-expressions) and [reference guide](https://github.com/justmarkham/DAT3/blob/master/code/99_regex_reference.py), [real-time regex tester](https://regex101.com/#python), [in-depth tutorials](http://www.rexegg.com/)
* [SpaCy](http://honnibal.github.io/spaCy/): a new NLP package


### Class 15: Decision Trees
* Decision trees ([IPython notebook](http://nbviewer.ipython.org/github/justmarkham/DAT4/blob/master/notebooks/15_decision_trees.ipynb))

**Homework:**
* By next Wednesday (before class), review the project drafts of your two assigned peers according to [these guidelines](peer_review.md). You should upload your feedback as a Markdown (or plain text) document to the "reviews" folder of DAT4-students. If your last name is Smith and you are reviewing Jones, you should name your file `smith_reviews_jones.md`.

**Resources:**
* scikit-learn documentation: [Decision Trees](http://scikit-learn.org/stable/modules/tree.html)

**Installing Graphviz (optional):**
* Mac:
    * [Download and install PKG file](http://www.graphviz.org/Download_macos.php)
* Windows:
    * [Download and install MSI file](http://www.graphviz.org/Download_windows.php)
    * Add it to your Path: Go to Control Panel, System, Advanced System Settings, Environment Variables. Under system variables, edit "Path" to include the path to the "bin" folder, such as: `C:\Program Files (x86)\Graphviz2.38\bin`


### Class 16: Ensembling
* Ensembling ([IPython notebook](http://nbviewer.ipython.org/github/justmarkham/DAT4/blob/master/notebooks/16_ensembling.ipynb))

**Resources:**
* scikit-learn documentation: [Ensemble Methods](http://scikit-learn.org/stable/modules/ensemble.html)
* Quora: [How do random forests work in layman's terms?](http://www.quora.com/How-do-random-forests-work-in-laymans-terms/answer/Edwin-Chen-1)


### Class 17: Databases and MapReduce
* Learn the basics of databases [database code](code/17_sql.py)
* MapReduce basics [slides](slides/17_db_mr.pdf)
* MapReduce example in python [code](code/17_map_reduce.py)

**Resources:**
* [Forbes: Is it Time for Hadoop Alternatives?](http://www.forbes.com/sites/johnwebster/2014/12/08/is-it-time-for-hadoop-alternatives/)
* [IBM: What is MapReduce?](http://www-01.ibm.com/software/data/infosphere/hadoop/mapreduce/)
* [Wakari MapReduce IPython notebook](https://www.wakari.io/sharing/bundle/nkorf/MapReduce%20Example)
* [What Every Data Scientist Needs to Know about SQL](http://joshualande.com/data-science-sql/)
* [Brandon's SQL Bootcamp](https://github.com/brandonmburroughs/sql_bootcamp)
* SQL tutorials from [SQLZOO](http://sqlzoo.net/wiki/Main_Page) and [Mode Analytics](http://sqlschool.modeanalytics.com/)


### Class 18: Recommenders
* Recommendation Engines [slides](slides/18_recommendation_engines.pdf)
* Recommendation Engine Example [code](code/18_recommenders_class.py)

**Resources:**
* [The Netflix Prize](http://www.netflixprize.com/)
* [Why Netflix never implemented the winning solution](https://www.techdirt.com/blog/innovation/articles/20120409/03412518422/why-netflix-never-implemented-algorithm-that-won-netflix-1-million-challenge.shtml)
* [Visualization of the Music Genome Project](http://www.music-map.com/)
* [The People Inside Your Machine](http://www.npr.org/blogs/money/2015/01/30/382657657/episode-600-the-people-inside-your-machine) (23 minutes) is a Planet Money podcast episode about how Amazon Mechanical Turks can assist with recommendation engines (and machine learning in general).


### Class 19: Advanced scikit-learn
* Advanced scikit-learn ([code](code/19_advanced_sklearn.py))
    * Searching for optimal parameters: [GridSearchCV](http://scikit-learn.org/stable/modules/grid_search.html)
    * Standardization of features: [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    * Chaining steps: [Pipeline](http://scikit-learn.org/stable/modules/pipeline.html)
    * Regularized regression ([notebook](http://nbviewer.ipython.org/github/justmarkham/DAT4/blob/master/notebooks/19_regularization.ipynb)): [Ridge, RidgeCV, Lasso, LassoCV](http://scikit-learn.org/stable/modules/linear_model.html)
    * Regularized classification: [LogisticRegression](http://scikit-learn.org/stable/modules/linear_model.html)
    * Feature selection: [RFE, RFECV](http://scikit-learn.org/stable/modules/feature_selection.html)

**Homework:**
* Read this classic paper, which may help you to connect many of the topics we have studied throughout the course: [A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf).

**Resources:**
* Here is a longer example of [feature scaling](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/about_standardization_normalization.ipynb) in scikit-learn, with additional discussion of the types of scaling you can use.
* [Clever Methods of Overfitting](http://hunch.net/?p=22) is a classic post by John Langford.
* [Common Pitfalls in Machine Learning](http://danielnee.com/?p=155) is similar to Langford's post, but broader and a bit more readable.


### Class 20: Course Review
* [Data science review](https://docs.google.com/document/d/1XCdyrsQwU5OC5os7RHdVTEtS-tpHBbsoKKWLpYI6Svo/edit?usp=sharing)
* [Comparing supervised learning algorithms](https://docs.google.com/spreadsheets/d/15_QJXm6urctsbIXO-C_eXrsSffbHedio8z0E5ozxO-M/edit?usp=sharing)

**Resources:**
* [Choosing a Machine Learning Classifier](http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/): Edwin Chen's short and highly readable guide.
* [scikit-learn "machine learning map"](http://scikit-learn.org/stable/tutorial/machine_learning_map/): Their guide for choosing the "right" estimator for your task.
* [Machine Learning Done Wrong](http://ml.posthaven.com/machine-learning-done-wrong): Thoughtful advice on common mistakes to avoid in machine learning.
* [Practical machine learning tricks from the KDD 2011 best industry paper](http://blog.david-andrzejewski.com/machine-learning/practical-machine-learning-tricks-from-the-kdd-2011-best-industry-paper/): More advanced advice than the resources above.
* [An Empirical Comparison of Supervised Learning Algorithms](http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf): Research paper from 2006.
* [Getting in Shape for the Sport of Data Science](https://www.youtube.com/watch?v=kwt6XEh7U3g): 75-minute video of practical tips for machine learning (by the past president of Kaggle).
* [Resources for continued learning!](resources.md)


### Class 21: Project Presentations


### Class 22: Project Presentations
