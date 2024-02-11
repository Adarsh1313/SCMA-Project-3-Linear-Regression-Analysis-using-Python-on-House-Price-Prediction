#!/usr/bin/env python
# coding: utf-8

# **House Price Prediction using Linear Regression in Python**

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os
os.chdir('D:\Downloads\Stats Modelling Projects\Project 3')


# In[3]:


os.getcwd()


# In[6]:


dataset = pd.read_excel("HousePricePrediction.xlsx")

# Printing first 10 records of the dataset
dataset.head(10)


# In[7]:


dataset.shape


# In[8]:


dataset.info()


# #### Inference 
# ##### - MS Zoning, Exterior1st, BsmtFinSF2, TotalBsmtSF have a few missing values
# ##### - SalePrice (the target variable has a lot of missing values)

# In[9]:


obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))
 
int_ = (dataset.dtypes == 'int64')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))
 
fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# In[10]:


type(object_cols)


# In[11]:


object_cols


# ## Exploratory Data Analysis

# In[12]:


dataset.describe()


# In[35]:


dataset.isna().sum()


# In[36]:


dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())


# #### Inference 
# ##### - SalePrice missing data is being filled with the mean of existing values, since it is numerical data

# In[13]:


dataset.isna().sum()


# In[14]:


dataset.drop(['Id'], axis=1, inplace=True)


# In[15]:


new_dataset = dataset.dropna()


# In[16]:


new_dataset.info()


# In[17]:


df_final = dataset.select_dtypes(include=['int64', 'float64']).dropna()


# #### Inference 
# ##### - As we are doing regression, we're dropping all categorical data and keeping only numerical data

# In[18]:


df_final.info()


# In[19]:


df_final.shape


# #### Out of 1460 records, we'll take 70% as training data and 30% as test data

# In[ ]:


#split into train and test data
# training data => 70% of 1460 = 1022
# test data => (total - training data)
#            => 1460 - 1022 = 438


# In[68]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
 
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']
 
# Split the training set into 
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.70, test_size=0.30, random_state=0)


# In[69]:


#len(xtrain) # all the independant variables
len(X_train), len(X_valid)


# In[70]:


#len(ytrain) # it only contains dependant or target variable
len(Y_train), len(Y_valid)


# In[71]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print('R-Sqaure :',r2_score(Y_valid, Y_pred))
print('mean_absolute_percentage_error is :',mean_absolute_percentage_error(Y_valid, Y_pred))
print('mean_absolute_error is :',mean_absolute_error(Y_valid, Y_pred))


# In[72]:


# Access the coefficients and intercept
coefficients = model_LR.coef_
intercept = model_LR.intercept_

# Print the coefficients and intercept
print('Coefficients:', coefficients)
print('Intercept:', intercept)


# In[73]:


import statsmodels.api as sm 

# Create a linear regression model using statsmodels
model = sm.OLS(Y_train, X_train)

# Fit the model
results = model.fit()

# Print the summary
print(results.summary())


# ## Conclusion
# #### - R-squared shows = 0.911 shows of the total variation in the dependent variable 91% of the variation in the dependant variable is explained by the various independant variables square feet.
# #### - Since we have 7 independant variables, the R Squared tends to be high - therefore we'll infer to the Adj R-Squared which is also equal to 0.911
