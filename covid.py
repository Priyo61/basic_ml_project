#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('C:\\Users\\Priyo\\OneDrive\\Desktop\\corona.csv')


# In[2]:


# check dataset for first 5 rows
df.head(5)


# In[3]:


#checking dataset for last 5 rows
df.tail(5)


# In[4]:


#checking data shape how many rows and columns 
df.shape


# In[5]:


# checking dataset columns name
df.columns


# In[6]:


# started data preporocessing missing element or not
df.isnull().sum()


# In[7]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[8]:


#checking datatypes
df.dtypes


# In[9]:


df.head(2)


# In[10]:


# dataset is categorical by looking dataset so i have to do logistic classificaton


# In[11]:


# Here i have decoded categorical to numerical for model building
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['n_corona']=le.fit_transform(df['Corona'])
df.tail(2)


# In[12]:


# Here i have done some data are not proper formate so i have change 
import numpy as np
df.replace('TRUE','True',inplace=True)
df.replace('FALSE','False',inplace=True)
df=pd.DataFrame(df)
df.head(5)


# In[13]:


# Here again i have decode data from categorical to numerical
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dep=df['n_corona']
indep=df[['Cough_symptoms', 'Fever', 'Sore_throat',
       'Shortness_of_breath', 'Headache','Sex']]
indep.head(2)
dep.value_counts()  # finding value count for decoded columns 


# In[14]:


# above n_corona 0=negative, 2=Positive, and 1=other


# In[15]:


# Here i have convert all elemnt to string 
df['Cough_symptoms']=df['Cough_symptoms'].astype('str')
df['Fever']=df['Fever'].astype('str')
df['Sore_throat']=df['Sore_throat'].astype('str')
df['Shortness_of_breath']=df['Shortness_of_breath'].astype('str')
df['Headache']=df['Headache'].astype('str')
df['Sex']=df['Sex'].astype('str')
df.head()


# In[16]:


#Here i have covert remaing categorical to numerical for modelling
df['n_cough']=le.fit_transform(df['Cough_symptoms'])
df['n_fever']=le.fit_transform(df['Fever'])
df['n_sor']=le.fit_transform(df['Sore_throat'])
df['n_breath']=le.fit_transform(df['Shortness_of_breath'])
df['n_head']=le.fit_transform(df['Headache'])
df['gender']=le.fit_transform(df['Sex'])
df.head(3)


# In[17]:


df['n_corona'].value_counts() # checking value count for new converted columns


# In[18]:


df['n_cough'].value_counts()


# In[19]:


# check all columns name both categorical and numerical
df.columns


# In[20]:


# Definding dependent and Independent cariables for modelling logistic classification
dep=df['n_corona']
ind=df[['n_cough', 'n_fever', 'n_sor', 'n_breath',
       'n_head', 'gender']]


# In[21]:


y=dep
x=ind


# In[22]:


# Normalising/standardzing independent variable
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
new_x=sc.fit_transform(x)
new_x=pd.DataFrame(new_x)
new_x
new_x.shape


# In[23]:


# here i have split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(new_x,y,test_size=0.25)
x_train
x_test


# In[24]:


# modelling split data
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[25]:


# i have predic test independnet data
pre=model.predict(x_test)
Res = pd.DataFrame({"Actual":y_test,"prediction":pre})

Res = pd.concat((new_x,Res), axis=1)
Res = Res[Res["Actual"]>=0]
Res.head(20)



# In[26]:


# Here i am finding accuracy by compering actual data and prediction data
from sklearn.metrics import accuracy_score
accuracy_score(Res['Actual'],Res['prediction'])


# In[27]:


sns.displot(Res[['Actual', 'prediction']], kde=True)


# In[28]:


sns.kdeplot(Res[['Actual', 'prediction']])


# In[29]:


# show data visualization
import plotly.express as px
fig = px.scatter(Res,x="Actual",y="prediction")
fig.show()


# In[30]:


# Decision Tree classification

from sklearn.tree import DecisionTreeClassifier
decision_model = DecisionTreeClassifier()
decision_model.fit(x_train, y_train)


# In[31]:


prediction_tree = decision_model.predict(x_test)
accuracy_score(y_test,prediction_tree)


# In[32]:


# Random Forest classification

from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier()
                                            
                                    

RandomForest_model.fit(x_train, y_train)


# In[33]:


prediction_r = RandomForest_model.predict(x_test)
accuracy_score(y_test,prediction_r)


# In[34]:


# XGBoost  classification

get_ipython().system('pip install xgboost')


# In[35]:


from xgboost import XGBClassifier
XGB_model = XGBClassifier()

XGB_model.fit(x_train, y_train)


# In[36]:


prediction_xgb = XGB_model.predict(x_test)
accuracy_score(y_test,prediction_xgb)


# In[37]:


# K Nearest Neighbor classification

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 5)

knn_model.fit(x_train, y_train)


# In[38]:


prediction_knn = knn_model.predict(x_test)
accuracy_score(y_test,prediction_knn)


# # Why is your proposal important in today’s world? How predicting a disease accurately can improve medical treatment?  

# ANS:- Because it will contribute to health sector to enhance future condition of health for today's world by predicting accurate medical team know how to take care for future disease by knowing pattern of advance data of health problem.

# # How is it going to impact the medical field when it comes to effective screening and reducing health care burden. 
# 
# 

# Ans:-By collecting sample of health for accurate predict health deasese will enchance  medical team so that they can take respective measure for feature outcome to away from health care burden.

# # If any, what is the gap in the knowledge or how your proposed method can be helpful if required in future for any other disease.

# Ans:- Will collect  proper sample data of patient with consulting domain health department for feature prediction of health outcome so that health department can take proper step to protect and planning people away from respective disease.

# # observation : while i have done EDA i get now dataset is categorical show i need to predict covid positive or negative so i have done basic classification logistic regression technique to predict model so, i got good accuracy which will contribute to health department from feature outcome of health.
