
# coding: utf-8

# In[316]:


import pandas as pd
import numpy as np
import random as r
import math
import os
os.chdir('C:\loan')


# In[317]:


x_train = pd.read_csv('X_train.csv',header=None)


# In[331]:


cen = []

pi= [.2,.2,.2,.2,.2 ]

mu = []

sigma =[1,1,1,1,1]

nk = [0,0,0,0,0]

O=[]


# In[332]:


def F_dist(x,m):
    sum = 0
    for i in range(len(x)):
        sum = sum + math.pow(x[i]-m[i],2)
    return math.sqrt(sum)


# In[333]:


def F_cen(c):
    return c.mean()


# In[334]:


def F_std(c):
    return c.std()


# In[335]:


cen= [list(x_train.iloc[np.random.randint(len(x_train))].values) for i in range(5) ] 


# In[336]:


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
        sigma[0] = list(F_std(cl_1))
    if(len(cl_2)>0):
        cen[1]=list(F_cen(cl_2))
        sigma[1] = list(F_std(cl_2))
    if(len(cl_3)>0):
        cen[2]=list(F_cen(cl_3))
        sigma[2] = list(F_std(cl_3))
    if(len(cl_4)>0):
        cen[3]=list(F_cen(cl_4))
        sigma[3] = list(F_std(cl_4))
    if(len(cl_5)>0):
        cen[4]=list(F_cen(cl_5))
        sigma[4] = list(F_std(cl_5))
    
    pd.DataFrame(cen).to_csv('centroids-'+str(i+1)+'.csv',index=False,header=None)


# In[337]:


mu= [list(x_train.iloc[np.random.randint(len(x_train))].values) for i in range(5) ] 


# In[338]:


mat = []
for i in range(11):
    mat.append([.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2])


# In[339]:


O=[]
for i in range(len(x_train)):
    O.append([.2,.2,.2,.2,.2])


# In[340]:


for itera in range(10):
    for j in range(5):
        row=[]
        s = 0.0
        for k in range (len(x_train)):
            e=0.0
            for i in range(len(x_train.columns)):
                e =  float(math.pow((x_train.iloc[k][i] - mu[j][i]),2)) / math.pow(float(sigma[j][i]),2)
                e = e * (-0.5)
                e = float(math.exp(e))
                e = float(e * pi[j] * (1 / float(sigma[j][i])))
            
            #row.append(round(e,2))
            O[k][j] = round(e,2) 
        
        for i in range(len(O)):
            O[i]=[item/sum(O[i]) for item in O[i]]

        pi[j]=sum([O[i][j] for i in range(len(x_train))])/len(x_train)
        nk[j]=sum([O[i][j] for i in range(len(x_train))])
        mu[j]= list((1/nk[j])* sum([O[i][j] * x_train.iloc[i] for i in range(len(x_train))]))
        for i in range(len(sigma[j])):
            for k in range(len(sigma[j])):
                if(i==k):
                    mat[i][k] = sigma[j][k] * sigma[j][k]
                else:
                    mat[i][k] = sigma[j][k] * sigma[j][k]
        pd.DataFrame(mat).to_csv('Sigma-'+str(j+1)+'-'+str(itera+1)+'.csv',index=False,header=None)
    
    pd.DataFrame(pi).to_csv('pi-'+str(itera+1)+'.csv',index=False,header=None)
    pd.DataFrame(mu).to_csv('mu-'+str(itera+1)+'.csv',index=False,header=None)
        #list((1/nk[j])* sum([O[i][j] * float(math.pow((x_train.iloc[i] - mu[j]),2)) for i in range(len(x_train))]))
    

