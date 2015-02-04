import re
import pandas as pd
import requests
import time
import csv
import matplotlib.pyplot as plt
import numpy as np

# change this to your pandora screen name!
screen_name = ''
#####


set_list = set()
stations = 	requests.get('http://pandorasongs.oliverzheng.com/username/'+screen_name).json()
for station in stations['stations'][2:4]:
	stationID = station['stationId']
	print "on "+ station['stationName']
	i = 0
	songs = requests.get('http://pandorasongs.oliverzheng.com/station/'+stationID+'/'+str(i)).json()
	while songs['hasMore']:
		for song in songs['songs']:
			set_list.add( song['link'] )
		i += 1
		songs = requests.get('http://pandorasongs.oliverzheng.com/station/'+stationID+'/'+str(i)).json()
set_list = list(set_list)
len(set_list)


# get attributes of each song
new_set_list = []
all_atributes = set()
for song in set_list:
    print song
    site_text = requests.get('http://pandora.com'+song).text
    attributes = [r.strip().replace('<br>','') for r in re.findall('\\t\\t\\t[\w\s]*<br>\\n', site_text) if len(r) >= 9]
    if len(attributes):
        new_set_list.append( {'name':song, 'attributes':attributes} )
        [all_atributes.add(a) for a in attributes]
        


#create and save dataframe
rows = []
all_atributes =  list(all_atributes)
for n in new_set_list:
    rows.append( [n['name']] + [a in n['attributes'] for a in all_atributes] )
df = pd.DataFrame(rows, columns = ['name']+list(all_atributes))
# save
df.to_csv('songs.csv')

