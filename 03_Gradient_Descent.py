# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:38:32 2022

@author: ParkHK
"""
import numpy as np
from matplotlib import pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=1.0

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
    
#dLoss/dw
def gradient(x,y):
    return 2*x*(x*w-y)

print("Predict (before training) ",4,forward(4))

epoch_list=[]
loss_list=[]
for epoch in range(100):
    for x_val,y_val in zip(x_data,y_data):
        grad=gradient(x_val,y_val)
        w=w-0.01*grad
        print("\tgrad: ",x_val,y_val,grad)
        l=loss(x_val,y_val)
    print("progress : ",epoch," w = ",w," loss = ",l)
    epoch_list.append(epoch)
    loss_list.append(l)
print("Predict (after training) ","4 hours", forward(4))

plt.plot(epoch_list,loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()