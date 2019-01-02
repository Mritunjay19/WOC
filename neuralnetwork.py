import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


#reading of data
f=pd.read_csv("F:\\WOC\\image1.csv",header=-1)
x=np.array(f)
[m,n]=np.shape(x)

#assigning random values 
#values of theta for hidden layer
theta1=np.array([rd.random()*10 for i in range(0,3)])
theta2=np.array([rd.random()*10 for i in range(0,3)])
theta3=np.array([rd.random()*10 for i in range(0,3)])
#values of theta for output unit
theta=np.array([rd.random()*10 for i in range(0,3)])
theta0=rd.random()*10

#dividing data in x1 , x2 and y 
x1=(x[:,0]-np.mean(x[:,0]))/np.ptp(x[:,0])
x2=(x[:,1]-np.mean(x[:,1]))/np.ptp(x[:,1])
y=x[:,2]-1
#learning rate
a=0.01
J=[]


#applying forward propagation -> backpropagation ->gradient descent
for i in range(0,100):
    
    #forward propagation
    z1=theta1[0]+np.dot(theta1[1],x1)+np.dot(theta1[2],x2)
    z2=theta2[0]+np.dot(theta2[1],x1)+np.dot(theta2[2],x2)
    z3=theta3[0]+np.dot(theta3[1],x1)+np.dot(theta3[2],x2)
    a1=1/(1+np.exp(-z1))
    a2=1/(1+np.exp(-z2))
    a3=1/(1+np.exp(-z3))
    
    z=theta0+theta[0]*a1+theta[1]*a2+theta[2]*a3
    h=1/(1+np.exp(-z))
    
    #cost function 
    J=np.append(J,-(np.sum(y*np.log(h)+(1-y)*np.log(1-h)))/m)
    
    #applying backpropagation
    delta=np.array([[0.0 for p in range(0,3)] for l in range(0,3)])
    del3=h-y
    delta2=np.array([0.0 for p in range(0,4)])

    for j in range(0,m):
        a_2=np.array([a1[j],a2[j],a3[j]])
        d2=(theta*del3[j])*(a_2*(1-a_2))
        a_1=np.array([1,x1[j],x2[j]])
        delta+=np.dot(d2.reshape(3,1),a_1.reshape(1,3))
        delta2+=np.dot(del3[j],np.array([1,a1[j],a2[j],a3[j]]))
    delta=delta/m
    delta2=delta2/m
    
    #updating values of theta
    theta0=theta0-a*delta2[0]
    for j in range(0,3):
        theta1[j]=theta1[j]-a*delta[0,j]
        theta2[j]=theta2[j]-a*delta[1,j]
        theta3[j]=theta3[j]-a*delta[2,j]
        theta[j]=theta[j]-a*delta2[j+1]
    

plt.plot(J,np.arange(1,101))





         