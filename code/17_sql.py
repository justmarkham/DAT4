import sqlite3 as lite
# python package to interface with database files

# connect to a local database
con = lite.connect('../data/tweets.db')

# create a Cursor object
cur = con.cursor()    


# select everything from tweets
cur.execute('SELECT * from Tweets')
cur.fetchall()


# insert a record
cur.execute("INSERT INTO Tweets VALUES(9,'Investors are claiming that $TSLA will only continue to fall!', -.67)")
# need ot commit to make sure that all changes are done
con.commit()


# select only a few columns
cur.execute('SELECT Text, Id from Tweets')
cur.fetchall()

# select with an id
cur.execute('SELECT Text, Sentiment from Tweets WHERE Id = 5')
cur.fetchone()

# grab all tweets with negative sentiment
cur.execute('SELECT Text, Sentiment from Tweets WHERE Sentiment  < 0')
cur.fetchall()


# close the connection if we are done with it.
con.close()
