#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries

import pandas as pd #pip install pandas
import numpy as np  #pip install numpy


# In[2]:


#Taking the Dataset

dataset = pd.read_csv('DigitalAd_dataset.csv')


# In[3]:


print(dataset.shape)
print(dataset.head(5))


# In[4]:


#segregate dataset into x(independent variable) and y(dependent variable)

X = dataset.iloc[:,:-1].values  
Y = dataset.iloc[:,-1].values


# In[5]:


#splitting data into Train and Test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25,random_state=0)


# In[6]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)


# In[7]:


# Training the model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)


# In[8]:


# prediction for all Test Data

y_pred = model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[9]:


# checking the CONFUSION MATRIX and ACCURACY

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"Accuracy : {accuracy_score(y_test,y_pred)*100}%")


# In[10]:


# Predicting weather new customer with Age and Salary will he Buy or Not

age = int(input("Enter the Customer's Age:"))
sal = int(input("Enter the Customer's Salary:"))
newcustomer = [[age,sal]]
result = model.predict(sc.transform(newcustomer))
print(result)
if result == 1:
    print("Customer will Buy the Product")
else:
    print("Customer will not Buy the Product")


# In[ ]:




