#!/usr/bin/env python
# coding: utf-8

# ## **Linear Regression with Python Scikit Learn**
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ### **Simple Linear Regression**
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# ### Author: Harshitha Megharaj
# 
# ### Student Marks Prediction
# 
# 
# **Task-1**

# In[28]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 


# ### Importing Data

# In[18]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)


# In[19]:


data


# In[20]:


data.describe()


# ### Visualizing the Data

# In[21]:


data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ## Training and Testing

# In[4]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ## Modeling

# In[22]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()   
regressor.fit(X_train, y_train) 
Y_pred = regressor.predict(X_test) 


# ## Plotting Regression Graph

# In[24]:


plt.scatter(X, y)
plt.plot(X_test, Y_pred);
plt.show()


# In[25]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# ## Making Prediction

# In[13]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ## Predicting the marks of the student, if a student studied for 9.25hrs in a day

# In[26]:


predicted = [[9.25]]
result = regressor.predict(predicted)
result


# ## Evalutaing the model

# In[29]:


from sklearn import metrics  
Linear = metrics.mean_absolute_error(y_test, Y_pred)
print('Mean absolute Error:',Linear)


# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# ## Random Forest Regressor

# In[30]:


from sklearn.ensemble import RandomForestRegressor


# In[31]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)
Y_pred = regressor.predict(X_test)


# In[33]:


df=pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
df


# ## Mean Absolute Error

# In[38]:


from sklearn import metrics
Random_Forest=metrics.mean_absolute_error(y_test, Y_pred)
print('Mean Absolute Error:',Random_Forest)


# ## Summarized  Table

# In[39]:


df1=pd.DataFrame({'Model':['Linear','Random_Forest'],'Mean_Ansolute_Error':['4.18', '6.86']})
df1

