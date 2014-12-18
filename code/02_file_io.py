'''
CLASS: Reading and Writing Files in Python
'''

'''
Part 1: Reading files
Note: 'rU' mode (read universal) converts different line endings into '\n'
'''

# read the whole file at once, return a single string
f = open('../data/drinks.csv', 'rU')
f.read()        # one big string including newlines
f.read()        # empty string
f.close()

# read one line at a time (entire file does not have to fit into memory)
f = open('../data/drinks.csv', 'rU')
f.readline()    # one string per line (including newlines)
f.readline()    # next line
f.close()

# read the whole file at once, return a list of lines
f = open('../data/drinks.csv', 'rU')
f.readlines()   # one list, each line is one string
f.close()

# use list comprehension to duplicate readlines without reading entire file at once
f = open('../data/drinks.csv', 'rU')
[row for row in f]
f.close()

# use a context manager to automatically close your file
with open('../data/drinks.csv', 'rU') as f:
    [row for row in f]

# split on commas to create a list of lists
with open('../data/drinks.csv', 'rU') as f:
    [row.split(',') for row in f]

# use the built-in csv module instead
import csv
with open('../data/drinks.csv', 'rU') as f:
    [row for row in csv.reader(f)]

# use next to grab the next row
with open('../data/drinks.csv', 'rU') as f:
    header = csv.reader(f).next()
    data = [row for row in csv.reader(f)]


'''
Part 2: Writing files
Note: 'wb' mode (write binary) is usually the recommended option
'''

# write a string to a file
nums = range(5)
with open('nums.txt', 'wb') as f:
    for num in nums:
        f.write(str(num) + '\n')

# convert a list of lists into a CSV file
output = [['col1', 'col2', 'col3'], [4, 5, 6]]
with open('example.csv', 'wb') as f:
    for row in output:
        csv.writer(f).writerow(row)

# use writerows to do this in one line
with open('example.csv', 'wb') as f:
    csv.writer(f).writerows(output)
