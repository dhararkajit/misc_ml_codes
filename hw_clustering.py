
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as r
import math
import os
os.chdir('C:\loan')


# In[2]:


x_train = pd.read_csv('X_train.csv',header=None)


# In[3]:


pi =[]

cen = []

mu = []

sigma =[]


# In[4]:


cen= [list(x_train.iloc[np.random.randint(len(x_train))].values) for i in range(5) ] 


# In[5]:


pi= [.2,.2,.2,.2,.2 ]


# In[6]:


C=np.cov(np.matrix(x_train).T)


# In[7]:


COVAR = [C[i][i] for i in range(11)]


# In[8]:


def F_dist(x,m):
    sum = 0
    for i in range(len(x)):
        sum = sum + math.pow(x[i]-m[i],2)
    return math.sqrt(sum)


# In[9]:


def F_cen(c):
    return c.mean()


# In[10]:


cl_1 = pd.DataFrame()
cl_2 = pd.DataFrame()
cl_3 = pd.DataFrame()
cl_4 = pd.DataFrame()
cl_5 = pd.DataFrame()


# In[11]:


for i in range(10):
    cl_1 = pd.DataFrame()
    cl_2 = pd.DataFrame()
    cl_3 = pd.DataFrame()
    cl_4 = pd.DataFrame()
    cl_5 = pd.DataFrame()
    for j in range(len(x_train)):
        dist = {}
        for k in range(5):
            dist[k] = F_dist(x_train.iloc[j],cen[k])
        value , key = min((v,k) for k,v in dist.items())
       
        if(key == 0):
            cl_1 = cl_1.append(x_train.iloc[j])
        elif(key == 1):
            cl_2 = cl_2.append(x_train.iloc[j])
        elif(key == 2):
            cl_3 = cl_3.append(x_train.iloc[j])
        elif(key == 3):
            cl_4 = cl_4.append(x_train.iloc[j])
        elif(key == 4):
            cl_5 = cl_5.append(x_train.iloc[j])
    if(len(cl_1)>0):
        cen[0]=list(F_cen(cl_1))
    if(len(cl_2)>0):
        cen[1]=list(F_cen(cl_2))
    if(len(cl_3)>0):
        cen[2]=list(F_cen(cl_3))
    if(len(cl_4)>0):
        cen[3]=list(F_cen(cl_4))
    if(len(cl_5)>0):
        cen[4]=list(F_cen(cl_5))


# In[ ]:





# In[ ]:




