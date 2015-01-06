'''
HOMEWORK SOLUTION: Reading and Writing Files in Python
'''

'''
PART 1:
Read in drinks.csv
Store the header in a list called 'header'
Store the data in a list of lists called 'data'
Hint: you've already seen this code!
'''

import csv
with open('../data/drinks.csv', 'rU') as f:
    header = csv.reader(f).next()
    data = [row for row in csv.reader(f)]


'''
PART 2:
Isolate the beer_servings column in a list of integers called 'beers'
Hint: you can use a list comprehension to do this in one line
Expected output:
    beers == [0, 89, ..., 32, 64]
    len(beers) == 193
'''

beers = [int(row[1]) for row in data]


'''
PART 3:
Create separate lists of NA and EU beer servings: 'NA_beers', 'EU_beers'
Hint: you can use a list comprehension with a condition
Expected output:
    NA_beers == [102, 122, ..., 197, 249]
    len(NA_beers) == 23
    EU_beers == [89, 245, ..., 206, 219]
    len(EU_beers) == 45
'''

NA_beers = [int(row[1]) for row in data if row[5]=='NA']
EU_beers = [int(row[1]) for row in data if row[5]=='EU']


'''
PART 4:
Calculate the average NA and EU beer servings to 2 decimals: 'NA_avg', 'EU_avg'
Hint: don't forget about data types!
Expected output:
    NA_avg == 145.43
    EU_avg == 193.78
'''

NA_avg = round(sum(NA_beers) / float(len(NA_beers)), 2)
EU_avg = round(sum(EU_beers) / float(len(EU_beers)), 2)


'''
PART 5:
Write a CSV file called 'avg_beer.csv' with two columns and three rows.
The first row is the column headers: 'continent', 'avg_beer'
The second and third rows contain the NA and EU values.
Hint: think about what data structure will make this easy
Expected output (in the actual file):
    continent,avg_beer
    NA,145.43
    EU,193.78
'''

output = [['continent', 'avg_beer'], ['NA', NA_avg], ['EU', EU_avg]]
with open('avg_beer.csv', 'wb') as f:
    csv.writer(f).writerows(output)


'''
BONUS:
Learn csv.DictReader() and use it to redo Parts 1, 2, and 3.
'''

# Part 1
with open('../data/drinks.csv', 'rU') as f:
    data = [row for row in csv.DictReader(f)]

# Note: storing the header isn't actually useful for parts 2 and 3
# Also note: dictionaries are unordered, so don't rely on the ordering
header = data[0].keys()

# Part 2
beers = [int(row['beer_servings']) for row in data]

# Part 3
NA_beers = [int(row['beer_servings']) for row in data if row['continent']=='NA']
EU_beers = [int(row['beer_servings']) for row in data if row['continent']=='EU']
