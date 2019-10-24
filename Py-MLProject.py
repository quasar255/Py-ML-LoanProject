#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import libraries
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.ticker as ticker
from sklearn import preprocessing                     # Perprocessing
from sklearn.model_selection import train_test_split  # Split data test train
from sklearn.neighbors import KNeighborsClassifier    # Do classification training
from sklearn import metrics  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#load loan data
get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[6]:


df = pd.read_csv('loan_train.csv')
df.head()


# ### Convert to date time object 

# In[7]:


# convert date time
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[8]:


# check paidoff vs collection
df['loan_status'].value_counts()


# In[9]:


# install seaborn 
get_ipython().system('conda install -c anaconda seaborn -y')


# In[10]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[11]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[12]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[13]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[14]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[15]:


# gender 0,1 
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[16]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[17]:


df[['Principal','terms','age','Gender','education']].head()


# In[18]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[19]:


X = Feature
X[0:5]


# In[20]:


# define labels
y = df['loan_status'].values
y[0:5]


# In[21]:


# Data Standardization to give data zero mean and unit variance 
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[22]:


# Train test split. Again use sklearn
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[23]:


## CLASSIFICATION begins ##
k = 7

# Train Model  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

# Predict
yhat = neigh.predict(X_test)
yhat[0:5]


# In[24]:


# Evaluate model accuracy

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[25]:


# Find best K
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
print(mean_acc)
print(mean_acc.max())


# # Decision Tree

# In[26]:


# Train test split
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print ('Train set:', X_trainset.shape,  y_trainset.shape)
print ('Test set:', X_testset.shape,  y_testset.shape)


# In[27]:


# define classifier and its parameters
from sklearn.tree import DecisionTreeClassifier
dTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
dTree # it shows the default parameters
dTree.fit(X_trainset,y_trainset)


# In[28]:


# Predict
predTree = dTree.predict(X_testset)
#Compare prediction
print (predTree [0:5])
print (y_testset [0:5])


# In[29]:


# Evaluate model accuracy
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# # Support Vector Machine

# In[30]:


# Test tarin data split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[31]:


# Do classification SVM using rbf kernel
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# In[32]:


# Predict
yhatsvm = clf.predict(X_test)
yhatsvm [0:5]


# In[33]:


# Evaluation
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhatsvm)


# # Logistic Regression

# In[34]:


# test train data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[35]:


# Do regresion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[36]:


# Predict
yhatlr = LR.predict(X_test)
yhatlr


# In[37]:


# Model evaluation ising jaccard index
from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))


# ## Model Evaluation using NEW Test set

# In[38]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report


# In[39]:


# Download and load the test set:
get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[40]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[41]:


test_df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Feature1 = test_df[['Principal','terms','age','Gender','weekend']]
Feature1 = pd.concat([Feature1,pd.get_dummies(test_df['education'])], axis=1)
Feature1.drop(['Master or Above'], axis = 1,inplace=True)
Feature1.head()
X1 = Feature1


# In[42]:


# Predict kNN
yhat1 = neigh.predict(X1)
yhat1


# In[43]:


# Evaluate model accuracy kNN
y1 = test_df['loan_status'].values
print("Train set Accuracy: ", metrics.accuracy_score(y1, neigh.predict(X1)))
print("Test set Accuracy: ", metrics.accuracy_score(y1, yhat1))
print("f1 score: ", metrics.f1_score(y1, yhat1, average="macro"))
print("Jaccard: ", jaccard_similarity_score(y1, yhat1, normalize=True, sample_weight=None))
print (classification_report(y1, yhat1))


# In[44]:


# Decision Tree Predict
predTree1 = dTree.predict(X1)
#Compare prediction
print (predTree1 [0:5])
print (y1 [0:5])
# Evaluate model accuracy
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y1, predTree1))
print("f1 score: ", metrics.f1_score(y1, predTree1, average="macro"))
print("Jaccard: ", jaccard_similarity_score(y1, predTree1, normalize=True, sample_weight=None))
print (classification_report(y1, predTree1))


# In[45]:


# SVM Predict
yhatsvm1 = clf.predict(X1)
yhatsvm1 [0:5]


# In[46]:


# Evaluation
from sklearn.metrics import jaccard_similarity_score
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y1, yhatsvm1))
print("f1 score: ", metrics.f1_score(y1, yhatsvm1, average="macro"))
print("Jaccard: ", jaccard_similarity_score(y1, yhatsvm1, normalize=True, sample_weight=None))
print (classification_report(y1, yhatsvm1))


# In[47]:


#logistic regression Predict
yhatlr1 = LR.predict(X1)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y1, yhatlr1))
print("f1 score: ", metrics.f1_score(y1, yhatlr1, average="macro"))
print("Jaccard: ", jaccard_similarity_score(y1, yhatlr1, normalize=True, sample_weight=None))
yhatlr_prob1 = LR.predict_proba(X1)
print(yhatlr)
print("log loss: ", log_loss(y1, yhatlr_prob1))
print (classification_report(y1, yhatlr1))


# # Report

# In[ ]:


| Algorithm          | Jaccard   | F1-score | LogLoss |
|--------------------|-----------|----------|---------|
| KNN                | .741      | .63     | NA      |
| Decision Tree      | .741      | .63     | NA      |
| SVM                | .741      | .63     | NA      |
| LogisticRegression | .259      | .11     | 23.14  |


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




