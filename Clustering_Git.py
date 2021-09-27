#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A=pd.read_csv("C://Users//Pranita//DL_P//Notebooks//DATA/Mall_Customers.csv")


# In[2]:


A.head()


# In[3]:


A.columns,A.shape


# In[4]:


import seaborn as sb
import matplotlib.pyplot as plt

sb.pairplot(A)


# In[5]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
B=A[["Annual Income (k$)","Spending Score (1-100)"]]
#B=pd.DataFrame(A.apply(le.fit_transform),columns=A.columns)


# In[6]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=4)

model=km.fit(B)


# In[7]:


model.inertia_


# In[8]:


#Elbow Curve

WCSS=[]
k=range(1,40)
for i in k:
    from sklearn.cluster import KMeans
    km=KMeans(n_clusters=i)
    model=km.fit(B)
    WCSS.append(model.inertia_)


# In[9]:


plt.scatter(k,WCSS)
plt.plot(k,WCSS,c="red")
plt.xlabel("no. of clusters")
plt.ylabel("inertia")
plt.show()
#k=5


# In[13]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=5)
model=km.fit(B)
   


# In[14]:


plt.scatter(B["Annual Income (k$)"],B["Spending Score (1-100)"],c=model.labels_)
plt.show()


# In[15]:


#for Agglomerative Clustering

from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.spatial import distance_matrix

# Create distance matrix for point to point distance
C=pd.DataFrame(distance_matrix(B.values,B.values))

# Pass linkage matrix to dendogram function and plot it
linkage_matrix=linkage(C)

# Pass linkage matrix to dendogram function and plot it
dendrogram(linkage_matrix)

plt.show()


# In[ ]:




