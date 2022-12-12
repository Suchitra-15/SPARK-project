#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import Dataset

# In[2]:


iris = pd.read_csv("C:/Users/admin/Downloads/Iris.csv")
iris


# ### Data Inspecting

# In[3]:


iris


# In[4]:


iris.shape


# In[5]:


iris.head()


# In[6]:


iris.tail()


# In[7]:


iris.info()


# In[8]:


iris.describe()


# ### Data Cleaning

# #### Asign numbers to the species

# In[9]:


iris['Species'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2},inplace=True)
iris


# In[10]:


iris.isnull().sum()


# Droping Id column

# In[11]:


iris = iris.drop('Id', axis = 1)
iris


# # KMeans clustering

# In[12]:


from sklearn.cluster import KMeans


# #### Finding the optimal number of clusters

# #### elbow method/SSD

# In[13]:


x = iris.iloc[:,[0,1,2,3]].values
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init=10, random_state=1)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

wcss


# In[14]:


plt.plot(range(1,11),wcss, marker = 'o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# #### Silhouette analysis

# In[15]:


from sklearn.metrics import silhouette_score
range_n_clusters = range(2,11)
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = num_clusters, max_iter = 50)
    kmeans.fit(x)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(x, kmeans.labels_)
    print("For n_clusters = {0}, the silhouette score = {1}".format(num_clusters,silhouette_avg))


# By looking silhouette analysis and elbow method,
#    - we see in the elbow method the range is in between 0.0(2) and 2.5(4.5)
#    - from silhouette analysis highest silhouette score is 0.55 for 3 clusters 
#    - so let us take k=3

# #### Applying kmeans to the Dataset

# In[16]:


kmeans=KMeans(n_clusters=3,init="k-means++", n_init=10,max_iter=300,random_state=1)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)


# In[17]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c="blue",label="Setosa")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c="red",label="Versicolour")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c="yellow",label="Virginica")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
           s=100,c="black",label="Centroids")
plt.legend()
plt.show()


# #### From above graph it is clearly visualise that there are three clusters and their centroides are black in colour. 

# In[ ]:





# In[ ]:




