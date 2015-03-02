
"""
Class 18: Recommendation Engines
Jaccard Similarity
Modified KNN Algorithm
"""

import pandas as pd
from collections import Counter
#read in brands data
user_brands = pd.read_csv('../data/user_brand.csv')
# user_brands = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/user_brand.csv')

#look at count of stores
user_brands.Store.value_counts()

# Series of user IDs, note the duplicates
user_ids = user_brands.ID

# groupby ID to see what each user likes!
user_brands.groupby('ID').Store.value_counts()

# turns my data frame into a dictionary
# where the key is a user ID, and the value is a 
# list of stores that the user "likes"
brandsfor = {str(k): list(v) for k,v in user_brands.groupby("ID")["Store"]}

# try it out. User 83065 likes Kohl's and Target
brandsfor['83065']
  
# User 82983 likes many more!
brandsfor['82983']
  


########################
## Jaccard Similarity ##
########################
'''
The Jaccard Similarity allows us to compare two sets
If we regard people as merely being a set of brands they prefer
the Jaccard Similarity allows us to compare people

Example. the jaccard similarty between user 82983 and 83065 is .125
            because
            brandsfor['83065'] == ["Kohl's", 'Target']
            brandsfor['82983'] == ['Hanky Panky', 'Betsey Johnson', 'Converse', 'Steve Madden', 'Old Navy', 'Target', 'Nordstrom']

the intersection of these two sets is just set("Target")
the union of the two sets is set(['Target', 'Hanky Panky', 'Betsey Johnson', 'Converse', 'Steve Madden', 'Old Navy', 'Target', 'Nordstrom'])
so the len(intersection) / len(union) = 1 / 8 == .125

EXERCISE: what is the Jaccard Similarity 
          between user 82956 and user 82963?
# ANSWER == 0.3333333333

'''
brandsfor['82956'] # == ['Diesel', 'Old Navy', 'Crate & Barrel', 'Target']

brandsfor['82963'] # == ['Puma', 'New Balance', 'Old Navy', 'Target']


'''
EXERCISE: Complete the jaccard method below.
          It should take in a list of brands, and output the 
          jaccard similarity between them

This should work with anything in the set, for example
jaccard([1,2,3], [2,3,4,5,6])  == .3333333

HINT: set1 & set2 is the intersection
      set1 | set2 is the union

'''

def jaccard(first, second):
  first = set(first)
  second = set(second)
  # the line below should be changed
  # ANSWER
  return float(len(first & second)) / len(first | second)

# try it out!
brandsfor['83065'] # brands for user 83065
brandsfor['82983'] # brands for user 82983
jaccard(brandsfor['83065'], brandsfor['82983'])


jaccard(brandsfor['82956'], brandsfor['82963'])




#######################
### Our Recommender ###
#######################

'''
Our recommender will be a modified KNN collaborative algorithm.
Input: A given user's brands that they like
Output: A set (no repeats) of brand recommendations based on
        similar users preferences

1. When a user's brands are given to us, we will calculate the input user's
jaccard similarity with every person in our brandsfor dictionary

2. We will pick the K most similar users and recommend
the brands that they like that the given user doesn't know about

EXAMPLE:
Given User likes ['Target', 'Old Navy', 'Banana Republic', 'H&M']
Outputs: ['Forever 21', 'Gap', 'Steve Madden']
'''


given_user = ['Target', 'Old Navy', 'Banana Republic', 'H&M']

#similarty between user 83065 and given user
brandsfor['83065']
jaccard(brandsfor['83065'], given_user) 
# should be 0.2

'''
EXERCISE
    Find the similarty between given_user and ALL of our users
    output should be a dictionary where
    the key is a user id and the value is the jaccard similarity
{...
 '83055': 0.25,
 '83056': 0.0,
 '83058': 0.1111111111111111,
 '83060': 0.07894736842105263,
 '83061': 0.4,
 '83064': 0.25,
 '83065': 0.2,
 ...}
'''
# ANSWER
similarities = {k: jaccard(given_user, v) for k, v in brandsfor.iteritems()}

similarities

K = 5 #number of similar users to look at


# Now for the top K most similar users, let's aggregate the brands they like.
# I sort by the jaccard similarty so most similar users are first
# I use the sorted method, but because I'm dorting dictionaries
# I specify the "key" as the value of the dictionary
# the key is what the list should sort on
# so the most similar users end up being on top
most_similar_users = sorted(similarities, key=similarities.get, reverse=True)[:K]

# list of K similar users' IDs
most_similar_users

# let's see what some of the most similar users likes
brandsfor[most_similar_users[0]]

brandsfor[most_similar_users[3]]

# Aggregate all brands liked by the K most similar users into a single set
brands_to_recommend = set()
for user in most_similar_users:
    # for each user
    brands_to_recommend.update(set(brandsfor[user]))
    # add to the set of brands_to_recommend
    
    
brands_to_recommend
# UH OH WE HAVE DUPLICATES. Banana Republic, Old Navy, Target are all repeats.


# EXERCISE: use a set difference so brands_to_recommend only has
# brands that given_user hasn't seen yet

# ANSWER
brands_to_recommend = brands_to_recommend - set(given_user)

# without duplicates
brands_to_recommend


####################
##One Step Further##
####################

# We can take this one step further and caculate a "score" of recommendation
# We will define the score as being the number of times
# a brand appears within the first K users
brands_to_recommend = []
for user in most_similar_users:
    brands_to_recommend += list(set(brandsfor[user]) - set(given_user))

# Use a counter to count the number of times a brand appears
recommend_with_scores = Counter(brands_to_recommend)

# Now we see Gap has the highest score!
recommend_with_scores

###################
#### Item based ###
###################

'''
We can also define a similary between items using jaccard similarity.
We can say that the similarity between two items is the jaccard similarity
between the sets of people who like the two brands.

Example: similarity of Gap to Target is:
'''
# filter users by liking Gap
gap_lovers = set(user_brands['Gap' == user_brands.Store].ID)
old_navy_lovers = set(user_brands['Old Navy' == user_brands.Store].ID)

# similarty between Gap and Old Navy
jaccard(gap_lovers, old_navy_lovers)


guess_lovers = set(user_brands['Guess' == user_brands.Store].ID)
# similarty between Gap andGuess
jaccard(guess_lovers, gap_lovers)


calvin_lovers = set(user_brands['Calvin Klein' == user_brands.Store].ID)
# similarty between Gap and Calvin Klein
jaccard(calvin_lovers, gap_lovers)


