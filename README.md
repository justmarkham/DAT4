## DAT4 Course Repository

Course materials for [General Assembly's Data Science course](https://generalassemb.ly/education/data-science/washington-dc/) in Washington, DC (12/15/14 - 3/16/15). View student work in the [student repository](https://github.com/justmarkham/DAT4-students).

**Instructors:** Sinan Ozdemir and Kevin Markham. **Teaching Assistant:** Brandon Burroughs.

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
2/16: *No Class* | 2/18: [Decision Trees and Ensembles](#class-15-decision-trees-and-ensembles)<br>**Milestone:** First Draft
2/23: [Advanced scikit-learn](#class-16-advanced-scikit-learn) | 2/25: [Databases and MapReduce](#class-17-databases-and-mapreduce)
3/2: [Recommenders](#class-18-recommenders) | 3/4: [Course Review, Companion Tools](#class-19-course-review-companion-tools)<br>**Milestone:** Second Draft (Optional)
3/9: [TBD](#class-20-tbd) | 3/11: [Project Presentations](#class-21-project-presentations)
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
* [Python exercise](code/02_file_io_homework.py)
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
* Read through this excellent example of [data wrangling and exploration in Pandas](http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_04_wrangling.ipynb).

**Optional:**
* To learn more Pandas, review this [three-part tutorial](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/), or review these three excellent (but extremely long) notebooks on Pandas: [introduction](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_5-Introduction-to-Pandas.ipynb), [data wrangling](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_6-Data-Wrangling-with-Pandas.ipynb), and [plotting](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_7-Plotting-with-Pandas.ipynb).

**Resources:**
* For more on Pandas plotting, read the [visualization page](http://pandas.pydata.org/pandas-docs/stable/visualization.html) from the official Pandas documentation.
* To learn how to customize your plots further, browse through this [notebook on matplotlib](http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section2_4-Matplotlib.ipynb).
* To explore different types of visualizations and when to use them, [Choosing a Good Chart](http://www.extremepresentation.com/uploads/documents/choosing_a_good_chart.pdf) is a handy one-page reference, and Columbia's Data Mining class has an excellent [slide deck](http://www2.research.att.com/~volinsky/DataMining/Columbia2011/Slides/Topic2-EDAViz.ppt).


### Class 6: Numpy, Machine Learning, KNN


### Class 7: scikit-learn, Model Evaluation Procedures


### Class 8: Linear Regression


### Class 9: Logistic Regression, Preview of Other Models


### Class 10: Model Evaluation Metrics


### Class 11: Working a Data Problem


### Class 12: Clustering and Visualization


### Class 13: Naive Bayes


### Class 14: Natural Language Processing


### Class 15: Decision Trees and Ensembles


### Class 16: Advanced scikit-learn


### Class 17: Databases and MapReduce


### Class 18: Recommenders


### Class 19: Course Review, Companion Tools


### Class 20: TBD


### Class 21: Project Presentations


### Class 22: Project Presentations
