'''
Multi-line comments go between three  quotation marks.
You can use single or double quotes, but NOT BOTH AT ONCE.
'''

# One-line comments come after the pound symbol



##### BASIC DATA TYPES ######

x = 5               # creates an object
type(5)             # assigning it to a variable is not required
type(5.0)           # float
type('five')        # str , single or double quotes
type(True)          # bool
5 / 2               # integer division ignores remainder
5 / 2.0             # the presence of a decimal means float division


###### LISTS ######

nums = [5, 5.0, 'five']     # multiple data types
nums                        # print the list
type(nums)                  # check the type: list
len(nums)                   # check the length: 3
nums[0]                     # print first element
nums[2] = 6                 # replace a list element
nums                        # IT CHANGED
nums.append(7)              # list 'method' that changes the list
nums.remove(5)              # another list method
nums.append(5)
nums.append("apple")
nums.remove("apple")
sorted(nums)                # 'function' that does not modify the list
sorted(nums, reverse=True)  # optional argument
sum(nums)                   # returns sum of a list of numbers
'''
note that list methods are a part of lists, like append and remove
but sorted is a built in python method that TAKES IN a list
'''

# let's use a for loop to "iterate" over this list
days_of_the_week = ['Monday', "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", 'Sunday']

# list slicing
weekdays = days_of_the_week[0:5]    # weekdays, NOTE doesn't include the index 5
weekdays[0:2]
days_of_the_week[5:]     # weekends
days_of_the_week[:5]
days_of_the_week[0:7:3]  # the third element means every second element

##### CONDITIONALS ######

# we will use an if else to implement logic
x              # remember x?
if x  > 10 :
    print "x is more than 10 !"
    print "hooray"
elif x > 100:
    print "x is HUGE"
    
#as soon as one conditional is satisfied, it stops checking!


temperature = 20
if temperature <= 32:
    print "water is ice"
elif temperature > 32 and temperature < 212:
    print "water is liquid"
else:                        #implicity means else if temperature >=212
    print "water will boil"


###### DICTIONARIES ######
'''
have a key value structure,
like a dictionary! :)

dictionary = {
    key1: value1,
    key2: value2,
    ....
}
'''

sinan = {
    "age" : 23,
    "location" : "Baltimore",
    "gender": "Male",
    "occupation": "Professor/Entrepreneur"
}

sinan['age']
sinan['location']

deniz = {
    "age" : 29,
    "location" : "Washington, D.C.",
    "gender": "Female",
    'occupation': "Security Analyst"
}

people = [sinan, deniz] #is a list of myself

people[0] # is the dictionary thar represents sinan
people[0]['age']

'''
for loops are a way of "iterating" over an item, like a list!
but also dictionaries!
'''

# iterating over a list
for day in days_of_the_week: # the "day" variable can really be called anything
    print "the day is " + day

for person in people:
    print type(person)
    print str(person['age']) + " and lives in "+person['location']




sinan.items()             # produces a list
sinan.items()[0][0]

# NOW I can iterate over sinan.items()
# iterating over a dictionaries "items"
for element in sinan.items():
    print element[0]

for first, second in sinan.items():
    print second
    
# Let's filter out some days based on an if!
for day in days_of_the_week: # the "day" variable can really be called anything
    if len(day) <= 6: #will only print days with length of 6 or less
        print day      
        
##### LIST COMPREHENSIONS ######
        
        
import math				# import statement

flubber = [2, 5, 7, 4, 2, 5]
len(flubber)
[n**2 for n in flubber]    # list comprehension
[y / 2.0 for y in flubber] #note the 2.0 instead of 2
[math.sqrt(r) for r in flubber]
'''
note I used a different variable name for each one
because it doesn't matter!
'''

###### FUNCTIONS #######

def give_me_five():         # function definition ends with colon
    print "I'm going to give you five"    
    return 5                # indentation required for function body
print "gave you five"
give_me_five()              # prints the return value (5)
a_new_variable = give_me_five()        # assigns return value to a variable, doesn't print it


def calc(x=1, y=2, op = "add"):         # three arguments (without any defaults)
    if op == 'add':                     # conditional statement
        return x + y
    elif op == 'subtract':
        return x - y
    elif op == "multiply":
        return x * y
    else:
        print 'Valid operations: add, subtract, multiply'

calc(5, 3, 'add')
calc(5, 3, 'subtract')
calc(5, 3, 'multiply')
calc([1,2],[3,4], "add")
#calc("sinan",[1,2], "add")    # will not work
calc(5, 3)                    # defaults to add

calc()                        # defaults everything!


point_1 = (1, 2, 2)   #tuples have a paranthesis
point_2  = (3, 4, 5)

pairs = zip(point_1, point_2)
for x in pairs:
    print x
    print x[0]
    print x[1]

[x[0] for x in pairs]