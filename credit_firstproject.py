#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('C:\\Users\\Priyo\\OneDrive\\Desktop\\project1\\cred.csv')
df.head(2)
df1=pd.read_csv('C:\\Users\\Priyo\\OneDrive\\Desktop\\project1\\cred_l.csv')
df1.head(2)

data = df.merge(df1)
data.head(
         )


# In[2]:


data.shape


# In[3]:


data.columns


# In[4]:


data.dtypes


# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data.duplicated()


# # Exploratory Data Analysis (EDA)

# In[8]:


data.isnull().sum()


# In[9]:


# dropping occupation type which has many null values
data.drop('Type_Occupation', axis=1, inplace=True)


# In[10]:


# Checking duplicates in 'ID' column
len(data['Ind_ID']) - len(data['Ind_ID'].unique())


# In[11]:


d = data[data.duplicated()]
print(d)


# In[12]:


# Checking Non-Numerical Columns
cat_columns = data.columns[(data.dtypes =='object').values].tolist()
cat_columns


# In[13]:


# Checking Numerical Columns
data.columns[(data.dtypes !='object').values].tolist()


# In[14]:


# Checking unique values from Categorical Columns

for i in data.columns[(data.dtypes =='object').values].tolist():
    print(i,'\n')
    print(data[i].value_counts())
    print('-----------------------------------------------')


# In[15]:


data.isnull().sum()


# In[16]:


data.describe()


# In[17]:


data['Annual_income'].fillna(data['Annual_income'].mean(),inplace=True)
data.isnull().sum()


# In[18]:


data['GENDER'].mode()


# In[19]:


data['Annual_income']=data['Annual_income'].astype('int64')
data.dtypes


# In[20]:


# Converting 'Birthday_count' values from Day to Years
data['Birthday_count'] = round(data['Birthday_count']/-365,0)
data.rename(columns={'Birthday_count':'Age'}, inplace=True)


# In[21]:


# Converting 'Employed_days' values from Day to Years
data['Employed_days'] = abs(round(data['Employed_days']/-365,0))
data.rename(columns={'Employed_days':'YEARS_EMPLOYED'}, inplace=True) 


# In[22]:


data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[23]:


data['label'].value_counts()  # 0 is application approved and 1 is application rejected. 


# In[24]:


# checking if there are still duplicate rows in Final Dataframe
len(data) - len(data.drop_duplicates())


# In[25]:


data.dtypes


# In[26]:


print(data['GENDER'].value_counts())
print(data['GENDER'].mode())


# In[27]:


data['GENDER'].fillna('F',inplace=True)
data.isnull().sum()


# In[28]:


# feature select tion
final=data[['GENDER','Car_Owner','Propert_Owner','CHILDREN','Annual_income','Type_Income','EDUCATION','Marital_status','Housing_type',
           'Age','Family_Members','label','YEARS_EMPLOYED']]
final.shape


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (8,8))
sns.heatmap(final.corr(), annot=True)
plt.show()


# In[30]:


cat_columns = final.columns[(final.dtypes =='object').values].tolist()
cat_columns


# In[31]:


#Converting all Non-Numerical Columns to Numerical
from sklearn.preprocessing import LabelEncoder

for col in cat_columns:
        globals()['LE_{}'.format(col)] = LabelEncoder()
        final[col] = globals()['LE_{}'.format(col)].fit_transform(final[col])
final.head()  


# In[32]:


for col in cat_columns:
    print(col , "  : ", globals()['LE_{}'.format(col)].classes_)


# In[33]:


final.corr()


# In[34]:


x = final.drop(['label'], axis=1)
y = final['label']


# In[35]:


x.head()


# In[36]:


y.head()


# # Machine Learning Model

# In[37]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[38]:


# lOgistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[39]:


model=LogisticRegression()
model.fit(x_train, y_train)


# In[40]:


prediction = model.predict(x_test)


# In[41]:


Res = pd.DataFrame({"actual":y_test,"prediction":prediction})
Res = pd.concat((x,Res), axis=1)
Res = Res[Res["actual"]>=0]
Res


# In[42]:


accuracy_score(Res["actual"],Res["prediction"])


# In[43]:


Res['prediction'].value_counts()


# In[44]:


# Decision Tree classification

from sklearn.tree import DecisionTreeClassifier
decision_model = DecisionTreeClassifier()
decision_model.fit(x_train, y_train)


# In[45]:


prediction = decision_model.predict(x_test)
accuracy_score(y_test,prediction)


# In[46]:


# Random Forest classification

from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier()
                                            
                                    

RandomForest_model.fit(x_train, y_train)


# In[47]:


prediction_r = RandomForest_model.predict(x_test)
accuracy_score(y_test,prediction_r)


# In[48]:


# Support Vector Machine classification

from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(x_train, y_train)


# In[49]:


prediction_svc = svc_model.predict(x_test)
accuracy_score(y_test,prediction_svc)


# In[50]:


# K Nearest Neighbor classification

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 5)

knn_model.fit(x_train, y_train)


# In[51]:


prediction_knn = knn_model.predict(x_test)
accuracy_score(y_test,prediction_knn)


# In[52]:


# XGBoost  classification

get_ipython().system('pip install xgboost')


# In[53]:


from xgboost import XGBClassifier
XGB_model = XGBClassifier()

XGB_model.fit(x_train, y_train)


# In[54]:


prediction_xgb = XGB_model.predict(x_test)
accuracy_score(y_test,prediction_xgb)


# # Conclusion
# As we have seen that,random forest or XGBoost Model is giving highest accuracy of 0.9, hence we will use any of them Model for predicion

# # Why is your proposal important in today’s world? How predicting a good client is worthy for a bank?  
# 
# # :-From my deep analysis for customer it will away from financial risk of banking sector where as when we predict good client we have to proper and deep analyst for client history income and expenditure and he/her occupation all thier financial properties than to give good prediction outcome so, that banking sector are away from financial deficit of bank and to improve client access of particular bank.
# 
# 
# 
# 
# 
# 

# # How is it going to impact the banking sector? 
# 
# # :- By giving good prediction of client with deep analyst of client financial history avoiding financial deficit of bank so, that i will improve banking sector they know how to handel client in future giving credit card to different client.
# 
# 

# # If any, what is the gap in the knowledge or how your proposed method can be helpful if required in future for any bank in India.
# 
# # :- In today digitalization world many small financial sector or some new banking sector they offere many credit card without knowing client past financial structure this led to many banking sector fall in financial deficit account of bank because non repayment after using credit and timely not paying due of credit card amount.
# 
# #          My some proposal to bank is need collect proper client past financial structure and properties which they have monthly and yearly income and how much income are under saving and expenditure they have in monthy and annual number of family and maritial status and also if he or she is income earner and their age by deep analyst and will give good prediction to bank avoid financial risk or fraud from client to enhance banking sector and also it will improve trusty of client to bank.
# 
