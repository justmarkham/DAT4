# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18

@author: sinanozdemir
adapted from https://www.wakari.io/sharing/bundle/nkorf/MapReduce%20Example
"""
import re

def mapper(line, queue = None):
    result = []
    # remove leading and trailing whitespace
    line = line.strip()
    # remove odd symbols from the text
    line = re.sub('[!"§$%&/()=?*#()\[\],.<>:;~_-]',"", line)
    # split the line into words
    words = line.split(" ")
    # insert the cleaned words into the results list
    for word in words:
        result.append((word, 1))
    if queue:
        queue.put(result)
    return result
    # output is a list of (key, value) pairs

mapper("Hi everyone Hi Hi")
# [('Hi', 1), ('everyone', 1), ('Hi', 1), ('Hi', 1)]
# note that duplicates are expected
   
   
   
# the reducer function is very simple! All it will do is sup up similar values from sorted key value pairs
def reducer(key, values):
    print "Reducer result -> %s : %d" % (key, sum(values))


# the shuffle function gathers up the like key words
# once it gathers them up, it calls the reduce function!
def shuffle(words):
    # sorting the words
    sorted_keys = sorted(words)
    tmp=""
    val_list=[]
    for i in sorted_keys:
        if i[0]!=tmp and tmp!="":
            print tmp, val_list
            reducer(tmp,val_list)
            val_list=[]
            tmp=i[0]
            val_list.append(i[1])
        elif i[0]==tmp or tmp=="":
            tmp=i[0]
            val_list.append(i[1])
    # get the last key value pair
    print tmp, val_list
    # now reduce the new key value pair
    reducer(tmp,val_list)
    
shuffle([('Hi', 1), ('everyone', 1), ('Hi', 1), ('Hi', 1)])

    
sentences = ['hello big data big big big data ',
             'big data is the best',
             'big data is the best data big',
             'hello big data how are data',
             'big big big data',
             'data data big big']
# list of sentences to analyze   

# get the first sentence
first_sentence = sentences[0]

# map the first sentence
mapper(first_sentence)

#send the mapped sequence to the shuffler/reducer
shuffle(mapper(first_sentence))


# now do it for all of the sentences one by one
output_map =[]
for sentence in sentences:
    output_map +=mapper(sentence)

# total (key: value) pairs in one list
output_map

# call the shuffle function, which also calls the reduce function
shuffle(output_map)


