#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


from sklearn.preprocessing import StandardScaler   #this standard scaler will be used to standardize th data to a common range


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn import svm


# In[8]:


from sklearn.metrics import accuracy_score


# In[9]:


diabetes_data = pd.read_csv('diabetes_data.csv')


# In[10]:


diabetes_data.head()


# In[11]:


diabetes_data.shape


# In[12]:


diabetes_data.describe()


# In[13]:


diabetes_data.isnull().sum()


# In[14]:


diabetes_data['Outcome'].value_counts()


# In[15]:


diabetes_data.groupby('Outcome').mean()


# In[16]:


X = diabetes_data.drop(columns = 'Outcome', axis = 1)


# In[17]:


Y = diabetes_data['Outcome']


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


# standardize the data in a particular range because we see in our data set pregnancy ki value alag h aur bhi sab
#   parameters ki value m bhot difference hai therefore we standardize the data in a particular range so that ml 
# algorithm works easily and better prediction for this we use standard scaler function


# In[21]:


scaler = StandardScaler()


# In[22]:


scaler.fit(X)


# In[23]:


#now transform this data 


# In[24]:


X = scaler.fit_transform(X)
print(X)


# In[25]:


#so uur data convert it into a 0 and 1 form


# In[26]:


print(Y)


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2 , stratify = Y, random_state = 2)


# In[28]:


print(X.shape, X_train.shape, X_test.shape)


# In[29]:


classifier = svm.SVC(kernel = 'linear')


# In[30]:


classifier.fit(X_train, Y_train)


# In[31]:


# model eualuation = to check how many times a model is precticted corectly so find accuracy score on traning data


# In[32]:


X_train_prediction = classifier.predict(X_train)  #model prediction stored in this variable


# In[33]:


training_data_accuracy = accuracy_score(X_train_prediction,Y_train) #predict accuracy withthe original output Y_TRAIN


# In[34]:


print(training_data_accuracy)


# In[35]:


# NOW FIND THE ACCURACY SCORE ON TEST DATA


# In[36]:


X_test_prediction = classifier.predict(X_test)


# In[37]:


training_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[38]:


print(training_data_accuracy)


# In[ ]:




