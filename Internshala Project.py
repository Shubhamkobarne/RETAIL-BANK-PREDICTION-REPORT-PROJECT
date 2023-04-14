#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[5]:


train=pd.read_csv(r"C:\Users\Admin\Downloads\Data_Science_Project_Problem_Statement\train.csv")
test=pd.read_csv(r"C:\Users\Admin\Downloads\Data_Science_Project_Problem_Statement\test.csv")


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


train.describe()


# In[9]:


test.describe()


# In[10]:


train.info()


# In[11]:


test.info()


# In[12]:


train['subscribed'].value_counts()


# In[13]:


train['subscribed'].value_counts(normalize=True)


# In[14]:


train['subscribed'].value_counts().plot.bar()


# In[15]:


sn.distplot(train["age"])


# In[16]:


train['job'].value_counts().plot.bar()


# In[17]:


train['default'].value_counts().plot.bar()


# In[18]:


print(pd.crosstab(train['job'],train['subscribed']))

job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')


# In[19]:


print(pd.crosstab(train['default'],train['subscribed']))

default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')


# In[20]:


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[21]:


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# In[22]:


train.isnull().sum()


# In[23]:


target = train['subscribed']
train = train.drop('subscribed',1)


# In[24]:


train = pd.get_dummies(train)


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


lreg = LogisticRegression()


# In[34]:


lreg.fit(X_train,y_train)


# In[35]:


prediction = lreg.predict(X_val)


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


accuracy_score(y_val, prediction)


# In[38]:


from sklearn.tree import DecisionTreeClassifier


# In[39]:


clf = DecisionTreeClassifier(max_depth=4, random_state=0)


# In[40]:


clf.fit(X_train,y_train)


# In[41]:


predict = clf.predict(X_val)


# In[42]:


accuracy_score(y_val, predict)


# In[43]:


test = pd.get_dummies(test)


# In[44]:


test_prediction = clf.predict(test)


# In[49]:


submission1 = pd.DataFrame()


# In[50]:


submission1['ID'] = test['ID']
submission1['subscribed'] = test_prediction


# In[51]:


submission1['subscribed'].replace(0,'no',inplace=True)
submission1['subscribed'].replace(1,'yes',inplace=True)


# In[52]:


submission1.to_csv('submission1.csv', header=True, index=False)

