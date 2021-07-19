#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:16:30 2021

@author: raman
"""

#K means to find clusters in the data set
   
import pandas as pd    
import seaborn as sns

df=pd.read_excel("/Users/raman/Desktop/MISM 6212/Week 10/assignment8.xlsx")

df=df.drop('University name', axis=1)

import seaborn as sns
sns.pairplot(df)

#minmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df)

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

wcv=[] #empty list for within cluster variation
silk_score=[]

for i in range(2,11):
        km=KMeans(n_clusters=i,random_state=0)
        km.fit(scaled_df)
        wcv.append(km.inertia_)
        silk_score.append( silhouette_score(scaled_df,km.labels_))
        
#elbow method
import matplotlib.pyplot as plt
plt.plot(range(2,11),wcv)
plt.xlabel("no of clusters")
plt.ylabel("within clusters variations")


#silhoutte score
plt.plot(range(2,11),silk_score)
plt.xlabel("no of clusters")
plt.ylabel("within clusters variations")
plt.grid()


#kmeans for 4 clusters
km4=KMeans(n_clusters=4,random_state=0)
km4.fit(scaled_df)  #training/cluster are identified 

#lables
km4.labels_

#adding these labels back to the orignial df
df["labels"]=km4.labels_

#interpret it using pandas

#show all 18 columns
pd.options.display.max_columns=None

#cluster 0
df.loc[df["labels"]==0].describe()
#high number of applicants applied, high accepted rate, and high number of people enrolled

#cluster 1
df.loc[df["labels"]==1].describe()
#Low number of applicants who applied, low accepted rate, and high number of people enrolled

#cluster 2
df.loc[df["labels"]==2].describe()
#high number of applicants who applied, low accepted rate and low number of people enrolled

#cluster 2
df.loc[df["labels"]==3].describe()
#low number of applicants who applied, low accepted rate and low enrolled number of people enrolled 

#Ward method 
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
linked=linkage(scaled_df,method="ward")
dendrogram(linked)
plt.show()
#Using the ward method, the plot shows clearly defined 4 clusters, orange, green, red, and blue.  
#The 4 clusters match with the initial clusters we defined using the elbow and silhouette method. 


