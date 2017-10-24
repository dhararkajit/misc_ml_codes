import pandas as pd
import random as r
import numpy as np
import sys

x_train = pd.read_csv(str(sys.argv[3]), header =None)
y_train = pd.read_csv(str(sys.argv[4]), header =None)
x_test = pd.read_csv(str(sys.argv[5]), header =None)

lmda = int(sys.argv[1])
sigma = float(sys.argv[2])

dim = len(x_train.columns)

v1 = lmda * np.identity(dim) + x_train.T.dot(x_train)
v2 = np.linalg.inv(v1)
v3 = x_train.T.dot(y_train)
w_RR = v2.dot(v3)

f= open(r'wRR_'+sys.argv[1]+'.csv','wb')
for line in w_RR:
	f.write(str(round(line,2)).strip("[]"))
	f.write('\n')
f.close()


var1 = lmda * np.identity(dim) * sigma
mu_not=[]
sig_not=[]
mu_not1=[]
sig_not1=[]
ind=[]

for i in range(0,10):
	var2 = var1 + x_train.T.dot(x_train)
	var3 = np.linalg.inv(var2)
	var4 = x_train.T.dot(y_train)
	mu = var3.dot(var4)
	var5= (var1/sigma)+1/float(sigma)*x_train.T.dot(x_train)
	sig=np.linalg.inv(var5)
	for j in range(0,len(x_test)):
		temp = x_test.iloc[j].T.dot(mu)
		mu_not.append(temp)
		temp=sigma+x_test.iloc[j].dot(sig.dot(x_test.iloc[j].T))
		sig_not.append(temp)
	mu_not1.append(mu_not[sig_not.index(max(sig_not))])
	mu_not2 =pd.DataFrame(mu_not1[i],index =[0],columns =[0])
	mu_not=[]
	sig_not1.append(sig_not.index(max(sig_not)))
	sig_not=[]
	x_train = x_train.append(x_test.iloc[sig_not1[i]])
	y_train = y_train.append(mu_not2.iloc[0])
	x_train = x_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	x_test = x_test.drop([sig_not1[i]],axis=0)
	x_test = x_test.reset_index(drop=True)
	count=0
	for k in range(0,i):
		if(sig_not1[k] <=sig_not1[i]):
			count = count +1
	ind.append(sig_not1[i]+count)


f= open(r'active'+'_'+sys.argv[1]+'_'+sys.argv[2]+'.csv','wb')
for line in ind:
	f.write(str(line+1))
	f.write(',')
f.close()



f= csv.writer(open(r'probs_test.csv','wb'),delimiter=',' )

for i in range(len(x_test)-1):
	f.writerow([round(class_Prob['X'+str(i)+' class= '+str(k)],2) for k in range(len(k_class))])
		
