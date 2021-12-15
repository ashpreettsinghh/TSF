#!/usr/bin/env python
# coding: utf-8

# # Task1 : Prediction Using Supervised ML

# by Ashpreet Singh

# In[16]:


#Importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics  


# In[49]:


# Reading data from the given link
link = "http://bit.ly/w-data"
data = pd.read_csv(link)
print("Data has been imported")
data.head(10)


# In[50]:


# Plotting the data on graph
data.plot(x='Hours',y='Scores', style='o')
plt.title("Graph between no. of hours and their respective scores")
plt.xlabel('Number of hours')
plt.ylabel('Scores')
plt.show()


# In[19]:


#preparing the data
#Splitting testing and training data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 


# In[20]:


X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[51]:


#Training the data
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("TRAINIG DONE")


# In[22]:


#Plotting the regression line
l = regressor.coef_*X+regressor.intercept_


# In[52]:


plt.scatter(X,y)
plt.plot(X,l);
plt.show()


# In[53]:


#Making prediction
print(X_test)
y_pred = regressor.predict(X_test)


# In[31]:


#Comparing actual vs predicted data
dataf = pd.DataFrame({"Actual Score": y_test, "Predicted" : y_pred})
dataf


# In[35]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print(f"No of Hours = {hours}")
print(f"Predicted Score = {own_pred[0]}")


# In[36]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# ________________________________________________________________________

# 
