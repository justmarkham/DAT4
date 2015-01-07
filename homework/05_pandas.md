## Class 5 Homework: Pandas

Check out this excellent example of [data wrangling and exploration in Pandas](http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_04_wrangling.ipynb).

* Assignment:
    * Read through the entire IPython Notebook.
    * As you get to each code block, **copy it into your own Python script** and run the code yourself. Try to understand exactly how each line works. You will run into Python functions that you haven't seen before!
    * Explore the data on your own using Pandas. At the bottom of your script, write out (as comments) **two interesting facts** that you learned about the data, and show the code you used to find those facts.
    * Create **two new plots** that show something interesting about the data, and save those plots as files. Include the plotting code at the bottom of your script.
    * Add your **Python script and image files** to your folder of DAT4-students and create a pull request.
* Tips:
    * You can ignore everything in the first code block except the three `import` statements.
    * Rather than downloading the data, you can read the file directly from its URL.
    * You may get a warning message a few times when running the code. The code still worked, so you can ignore the message (or you can figure out how to rewrite the code to not trigger the warning).
    * The plotting code calls matplotlib directly instead of using Pandas plotting, but the effect is the same:
        * matplotlib: `plt.hist(data.year, bins=62)`
        * Pandas: `data.year.hist(bins=62)`
    * In the plotting code, don't run `remove_border()`.
    * Don't worry about running (or trying to understand) the code in the "Small Multiples" section.
