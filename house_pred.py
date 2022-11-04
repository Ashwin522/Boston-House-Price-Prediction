#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sklearn
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

import pandas as pd
import numpy as np
data = load_boston()
array = data.feature_names
print(array)
array = np.append(array,['medv'])
data, target = data.data, data.target
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,target,test_size=0.3)

print(Xtrain.shape,Ytrain.shape)

TARGET_PRICE = 'medv'

# instantiate the LabelEncoder
le = LabelEncoder()
df = pd.read_csv('./boston.csv')

y = df[TARGET_PRICE]
print(y.head())
df = df.drop([TARGET_PRICE], axis=1)

x = df
print(x.head())
dt = RandomForestRegressor(criterion='mae',n_jobs=-1, n_estimators=10,max_depth=6, min_samples_leaf=1, random_state=3)
dt.fit(Xtrain,Ytrain)
y_predicted = dt.predict(Xtest)
print(y_predicted)
accuracy = dt.score(Xtest,Ytest)
MSE_score = MSE(Ytest,y_predicted)

# print the final results

print("Testing Accuracy:",accuracy)
print("Mean Squared Error",MSE_score.mean())


# In[ ]:





# In[ ]:




