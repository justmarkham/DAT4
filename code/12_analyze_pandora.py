
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics


df = pd.read_csv('../data/songs.csv')


# perform clustering with 4 clusters
song_cluster = KMeans(n_clusters=4, init='random')
song_cluster.fit(df.drop('name', axis=1))
y_kmeans = song_cluster.predict(df.drop('name', axis=1))

# get info on one cluster
for cluster_in_question in range(0,4):
    # get center of cluster
    song_cluster.cluster_centers_[cluster_in_question]
    # grab songs in dataframe that belong to this cluster
    print df[np.where(y_kmeans == cluster_in_question, True, False)]['name']
    # look at top five qualities in cluster
    print sorted(zip(df.columns[1:], song_cluster.cluster_centers_[cluster_in_question]), key=lambda x:x[1], reverse=True)[1:6]
 
metrics.silhouette_score(df.drop('name',axis=1), song_cluster.labels_, metric='euclidean')   
    
# perform k means with up to 15 clusters
k_rng = range(1,15)
est = [KMeans(n_clusters = k).fit(df.drop('name',axis=1)) for k in k_rng]



# calculate silhouette score
from sklearn import metrics
silhouette_score = [metrics.silhouette_score(df.drop('name',axis=1), e.labels_, metric='euclidean') for e in est[1:]]

# Plot the results
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.plot(k_rng[1:], silhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.plot(4,silhouette_score[2], 'o', markersize=12, markeredgewidth=1.5,
markerfacecolor='None', markeredgecolor='r')
