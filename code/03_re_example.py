"""
This is an intro to regular expressions

I use https://regex101.com/#python to check my work!
"""


import re

# flow:
# create a re pattern object
# search (or match) it against text
# orgnize the captures patterns in groups

# \d matches a number
text = "Hello! My name is Sinan. It is 2014 and it's amazing."
pattern1 = re.compile("\d")
re.search(pattern1, text) # == a search object
# use group to get each instance in the regular expression
# \d is just ONE number, so it only finds the "2" in "2014"
re.search(pattern1, text).group(0)


# adding a + means "at least one" but potentially more
pattern2 = re.compile("\d+")
re.search(pattern2, text).group(0) # == '2014'

# use square brackets [] to match one of the items present
alphabet = 'abcdefg'
pattern3 = re.compile('[cfg]')
re.search(pattern3, alphabet).group(0)

mystery_pattern = re.compile("\d+-\d+-\d+")
# take a few minutes, and discuss, what application could this mystery_pattern have
re.search(mystery_pattern, "my phone number is 609-462-6706 dude").group(0)


# . matches ANYTHING
all_of_the_text = "dmzhvbekuhvbc     dfljghwco87rc6geinsr6t4gi7rgwefiuvbekuhvbdfljghwco87rc6geinsr6t4gi7rgwefiu ywgsfybcstzvgbrtybte"
anything_pattern = re.compile(".+")
re.search(anything_pattern, all_of_the_text).group(0)

# \w matches any word character, alphanumeric
# if you want to match an actual period, do \.
email_pattern = re.compile("[\w\.]+@\w+\.com")
re.search(email_pattern, "my email address is sinan.u.ozdemir@gmail.com").group(0)
