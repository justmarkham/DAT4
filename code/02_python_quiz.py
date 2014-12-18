'''
Python quiz (no Googling!)
'''

## SECTION 1:

# 1. How do you create an empty list named mylist?

# 2. Without running the code, what will this return?
5 == 5.0

# 3. Without running the code, what will this return?
5 > 3 or 5 < 3

# 4. How do I return the number 50 from this list?
numlist = range(1, 101)

# 5. Without running the code, what will these return?
len(numlist)
type(numlist)



## SECTION 2:

# 6. Divide 3 by 2 and get a result of 1.5

# 7. Import the math module, and use the sqrt() function to take the square root of 1089.

# 8. Slice newlist to return a list with the numbers 2, 3, 4:
newlist = [0, 1, 2, 3, 4, 5, 6]

# 9. From this dictionary, return the 10, and then change the 30 to a 40.
d = {'a':10, 'b':20, 'c':30}

# 10. Convert this for loop into a "better" for loop that doesn't use the i variable:
fruits = ['apple', 'banana', 'cherry']
for i in range(len(fruits)):
    print fruits[i].upper()



## SECTION 3:

# 11. Define a function "calc" that takes two variables, a and b, and returns their sum.

# 12. Join these two lists to make a single list [1, 2, 3, 4]:
list1 = [1, 2]
list2 = [3, 4]

# 13. Return Brandon's state only:
locations = {'Sinan': ['Baltimore', 'MD'], 'Brandon': ['Arlington', 'VA']}

# 14. Without running the code, what will the second line return?
nums = [1, 2, 3]
[num*2 for num in nums]

# 15. Turn this for loop into a list comprehension:
upperfruits = []
for fruit in fruits:
    upperfruits.append(fruit.upper())
