#Importing the necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#Storing it in a variable & reading the dataset

df = pd.read_csv("D:\AIML\Dataset\FuelConsumption.csv")

df.head()


# In[13]:


cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]

cdf


# In[5]:


#Splitting the data

msk = np.random.rand(len(cdf)) <= 0.80


# In[6]:


train = cdf[msk]
test = cdf[~msk]


# In[7]:


#Plotting the train dataset to check linearity

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,c='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


#Creating IV & DV & storing it in a variable

x = np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
y = np.asanyarray(train[["CO2EMISSIONS"]])


# In[21]:


x[:,2]


# In[30]:


#Importing necessary packages

from sklearn import linear_model


# In[31]:


regr = linear_model.LinearRegression()


# In[32]:


regr.fit(x,y)


# In[33]:


regr.coef_


# In[34]:


regr.intercept_


# In[36]:


#Plotting the graph

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,c='black')
plt.plot(x, regr.coef_[0][0]*x + regr.intercept_, c='Orange')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[37]:


#Plotting the graph

plt.scatter(train.CYLINDERS,train.CO2EMISSIONS,c='black')
plt.plot(x,regr.coef_[0][1]*x + regr.intercept_,c='red')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[38]:


plt.scatter(train.FUELCONSUMPTION_COMB,train.CO2EMISSIONS,c='black')
plt.plot(x,regr.coef_[0][2]*x + regr.intercept_,c='red')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[39]:


#Creating IV & DV for test dataset & storing it in a variable

test_x = np.asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])


# In[40]:


predicted_y = regr.predict(test_x)


# In[ ]:


#test_y = np.asanyarray(test_y)


# In[ ]:


#test_y


# In[41]:


print(f"Mean Absolute Error = {np.mean(np.absolute(test_y-predicted_y))}")


# In[42]:


print(f"Mean Square Error = %.2f" % np.mean((test_y-predicted_y)**2))


# In[43]:


from sklearn.metrics import r2_score


# In[44]:


print(f"R2-Score = {r2_score(predicted_y,test_y)*100} %")


# In[ ]:




