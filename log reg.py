import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

class logreg:
    
    #reading data
    def __init__(self):
        f=pd.read_csv("F:\WOC\LogisticRegressionData1.csv",header=-1)
        self.d=np.array(f.values)
    
    #adding additional column containing 1 + mean normalisation
    def normalise(self):
        mn=[]
        rg=[]
        x=[]
        [m,n]=np.shape(self.d)
        for j in range(0,n-1):
            mn=np.append(mn,np.mean(self.d[:,j]))
            rg=np.append(rg,np.max(self.d[:,j])-np.min(self.d[:,j]))
        for i in range(0,m):
            x=np.append(x,1.0)
            for j in range(0,n-1):
                x=np.append(x,(self.d[i,j]-mn[j])/rg[j])
            x=np.append(x,self.d[i,n-1])
        return x.reshape(m,n+1)
    
    
    #setting values of theta
    def val_theta(self,n):
        th=[]
        for i in range(0,n):
            th=np.append(th,rd.randint(1,100))
        return np.vstack(th)
    
    
    #applying gradient descent
    def gradient_des(self,x,y,th,m,n):
        J=[]
        t=np.empty(n)
        a=0.01
        for i in range(0,100):
            z=np.dot(x,th)
            z=np.exp(-z)
            h=1/(1+z)
            
            J=np.append(J,-np.sum(y*np.log(h)+(1-y)*np.log(1-h))/m)
            
            for j in range(0,n):
                x1=np.vstack(x[:,j])
                t[j]=th[j]-a*(np.sum((h-y)*x1)/m)
            for j in range(0,n):
                th[j]=t[j]
        return th
            
    


a=logreg()
data=a.normalise()
[m,n]=np.shape(data)
theta=a.val_theta(n-1)
theta=a.gradient_des(data[:,0:n-1],np.vstack(data[:,n-1]),theta,m,n-1)
print(theta)