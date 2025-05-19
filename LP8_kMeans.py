import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeans:
  def __init__(self,n_clusters,max_iterations):
    self.n_clusters=n_clusters
    self.max_iterations=max_iterations
    self.clusters=None
    self.centroids=None

  def fit(self,data):
    self.centroids=data[np.random.choice(len(data),self.n_clusters,replace=False)]

    for _ in range(self.max_iterations):
    # THIS IS THE MAIN LOOP
      clusters=[]
      for point in data:
        distances=[np.sqrt(np.sum((point-c)**2)) for c in self.centroids]
        closest_centroid=np.argmin(distances)
        clusters.append(closest_centroid)

      new_centroids=[]

      for i in range(self.n_clusters):
        #i in range of number of clusters
        cluster_points=data[np.array(clusters)==i]
        if len(cluster_points)>0:
          new_centroids.append(cluster_points.mean(axis=0))
        else:
          new_centroids.append(self.centroids[i])

      if np.allclose(self.centroids,new_centroids):
        break
      self.centroids=new_centroids
    self.clusters=clusters

data=pd.read_csv('/content/iris_csv (1).csv')
X=data.iloc[:,2:4]
X=np.float64(X)
model=KMeans(n_clusters=3,max_iterations=100)
model.fit(X)

plt.scatter(X[:,0],X[:,1],c=model.clusters)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('K Means Clustering')
plt.show()
