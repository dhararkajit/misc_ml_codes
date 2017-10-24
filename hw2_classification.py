import pandas as pd
import random as r
import numpy as np
import sys
import math
import csv


x_train = pd.read_csv(str(sys.argv[1]), header =None)
y_train = pd.read_csv(str(sys.argv[2]), header =None)
x_test = pd.read_csv(str(sys.argv[3]), header =None)

dim = len(x_train.columns)
k_class = list(y_train[0].unique())


class_Prob={}
class_Prob2={}
prob={}
mu_dict={}
std_dict = {}
ind =[]
e =0
row = []




for k in range(len(list(k_class))):
    prob[k] = len(y_train[y_train[0] == k_class[k]]) / float(len(y_train))

for i in range(len(list(k_class))):
	ind.append(y_train[y_train[0] == k_class[i]].index)
	temp=list(x_train.iloc[ind[i]].mean())
	mu_dict[i] = temp
	temp=[]
	temp=(list(x_train.iloc[ind[i]].var()))
	std_dict[i] = temp
	temp=[]



for j in range(len(x_test)-1):
	row=[]
	for k in range (len(k_class)):
		e=0.0
		for i in range(len(x_train.columns)):
			e =  float(math.pow((x_test.iloc[j][i] - mu_dict[k][i]),2)) / float(std_dict[k][i])
			e = e * (-0.5)
			e = float(math.exp(e))
			e = float(e * prob[k] * (1 / math.sqrt(float(std_dict[k][i]))))
		
		row.append(round(e,2))

	class_Prob[j]=row


for i in range(len(class_Prob)):
	row=[]
	for j in range(len(class_Prob[i])):
		row.append(round(class_Prob[i][j]/sum(class_Prob[i]),2))
	class_Prob[i] = row


output = pd.DataFrame.from_dict(class_Prob,orient='index')

output.to_csv('probs_test.csv',index=False,header=None)


	