
# coding: utf-8

# # 导入包

# In[58]:


import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
pd.set_option('display.max_rows',None)

pd.options.display.max_columns = 99

# 导入数据

os.chdir('/home/weifeng/learngit/泰坦尼克号生还者分析')
train = pd.read_csv("train.csv")
test =pd.read_csv("test.csv")
data = train.append(test)

## title from name

#data = train.append(test)
data['Title'] = data['Name']
for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.',expand = True)
data['Title'] = data['Title'].replace(['Capt', 'Col', 'Dona', 'Dr','Jonkheer','Major', 'Rev', 'Sir','Don'], 'Male_Rare')

data['Title'] = data['Title'].replace(['Countess', 'Lady','Mlle', 'Mme', 'Ms'], 'Female_Rare')
#data.Title.value_counts()

##  famliy size

data['family_size']  = data['SibSp'] + data['Parch'] + 1

#from sklearn.preprocessing import OneHotEncoder, LabelEncoder
bins = [0,1,4,20]
data['family_group'] = pd.cut(data['family_size'],bins)
label = LabelEncoder()
data['family_group'] = label.fit_transform(data['family_group'])

data['family_group']


## fare

data['Fare'].fillna(data['Fare'].dropna().median(), inplace=True)
#fare_bins = [0,4,10,20,45,600]
# data['fare_group'] = pd.cut(data['Fare'],fare_bins)
#print(data['fare_group'])
#label = LabelEncoder()
data.loc[data['Fare'] <= 4, 'Fare'] = 0
data.loc[(data['Fare'] > 4) & (data['Fare'] <= 10), 'Fare'] = 1
data.loc[(data['Fare'] > 10) & (data['Fare'] <= 20), 'Fare'] = 2
data.loc[(data['Fare'] > 20) & (data['Fare'] <= 45), 'Fare'] = 3
data.loc[data['Fare'] > 45, 'Fare'] = 4
data['fare_group'] = label.fit_transform(data['Fare'])
data['fare_group']


# Embarked

data['Embarked'] = data['Embarked'].fillna('S')


#print(data.info())

##  Age

data = data.drop(['Cabin','family_size'],axis = 1)
train = data[:891]
test = data[891:]

train = pd.get_dummies(train, columns=['Title','Sex',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)
test = pd.get_dummies(test, columns=['Title','Sex',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)


train.drop([ 'Ticket','Name', 'Fare','PassengerId'], axis=1, inplace=True)

test.drop([ 'Ticket','Name', 'Fare','Survived','PassengerId'], axis=1, inplace=True)

pd.options.display.max_columns = 99
test.head()


##  补全age

#age_df  = train.copy()
#age_df1 = test.copy()
#print(train.info())
#age_df.drop([ 'Age'], axis=1, inplace=True)
#age_df1.drop([ 'Age'], axis=1, inplace=True)
#print(train.info())
#age_df_notnull = age_df.loc[(train['Age'].notnull())] 
#age_df1_notnull = age_df1.loc[(test['Age'].notnull())]
#age_df_isnull = age_df.loc[(train['Age'].isnull())]   
#age_df1_isnull = age_df1.loc[(test['Age'].isnull())] 
#X = age_df_notnull.values[:,1:] 
#x = age_df1_notnull.values[:,1:]
#Y = age_df_notnull.values[:,0]                             
#y = age_df1_notnull.values[:,0]
# use RandomForestRegression to train data                 
#RFR = RandomForestRegressor(n_estimators=80, n_jobs=-1)                   
#RFR1 = RandomForestRegressor(n_estimators=80, n_jobs=-1) 
#RFR.fit(X,Y)                                               
#RFR1.fit(x,y)
#predictAges = RFR.predict(age_df_isnull.values[:,1:])      
#predictAges1 = RFR1.predict(age_df1_isnull.values[:,1:])
#train.loc[train['Age'].isnull(), ['Age']]= predictAges
#test.loc[test['Age'].isnull(), ['Age']]= predictAges1
#print(test.info())
#age_bins = [0,1,4,13,18,35,45,55,65,180]
#train['age_group'] = pd.cut(train['Age'],age_bins)
#train['age_group']
#train['Age']
from fancyimpute import KNN
age_train = KNN(10).complete(train)
train = pd.DataFrame(age_train,columns = train.columns)
#printtrain['Age']
age_test = KNN(k=10).complete(test)

test = pd.DataFrame(age_test, columns = test.columns)

age_bins = [0,1,4,13,18,35,45,55,65,180]
train['age_group'] = pd.cut(train['Age'],age_bins)
label = LabelEncoder()
train['age_group'] = label.fit_transform(train['age_group'])

test['age_group'] = pd.cut(test['Age'],age_bins)
test['age_group'] = label.fit_transform(test['age_group'])
train.head()

train = pd.get_dummies(train,columns=['age_group'], drop_first=True)

test = pd.get_dummies(test,columns=['age_group'], drop_first=True)
print(train.head())
train = train.drop(['Age','Parch','SibSp'],axis=1)
test = test.drop(['Age','Parch','SibSp'],axis=1)


x = train.drop(['Survived'],axis = 1)


y = train['Survived']


##  XGB

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(x,y)
gbc_pred = gbc.predict(test).astype(int)
XGBClassifier = XGBClassifier()
XGBClassifier.fit(x, y)
y_pred = XGBClassifier.predict(test).astype(int)

passengers = np.arange(892,1310)
name1 = ['PassengerId']
name2 =['Survived']
columns1 = pd.DataFrame(columns = name1,data = passengers)
columns2 = pd.DataFrame(columns = name2,data = gbc_pred)
results = columns1.join(columns2)
results = results.set_index('PassengerId')

results.to_csv('new4.csv')


# In[56]:


train.head()

