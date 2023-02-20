from k_means import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

X = df[['SL', 'PL']]

model = KMeans(max_iter = 500, tolerance = 0.001, n_clusters = 3, runs = 100)
(clusters, data_with_clusters) = model.fit(X)

plt.close()

#number of clusters
K=3

#Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["SL"],X["PL"],c='black')
plt.scatter(Centroids["SL"],Centroids["PL"],c='red', marker='*', s=200, edgecolors="k")
plt.xlabel('PL')
plt.ylabel('SL')
plt.show()

diff = 1
j=0
suma = 0.01

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["SL"]-row_d["SL"])**2
            d2=(row_c["PL"]-row_d["PL"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["PL","SL"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['PL'] - Centroids['PL']).sum() + (Centroids_new['SL'] - Centroids['SL']).sum()
        suma = diff.sum()
        print(suma)
    Centroids = X.groupby(["Cluster"]).mean()[["PL","SL"]]

    color = ['blue', 'green', 'orange']
    for k in range(K):
        data = X[X["Cluster"] == k + 1]
        plt.scatter(data["SL"], data["PL"], c=color[k])
    plt.scatter(Centroids["SL"], Centroids["PL"], c='red', marker='*', s=200, edgecolors="k")
    plt.xlabel('SL')
    plt.ylabel('PL')
    if(suma == 0):
        plt.title("BAIGTA")
    plt.show()