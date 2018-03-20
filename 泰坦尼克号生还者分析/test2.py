import os		
os.chdir('/home/weifeng/learngit/泰坦尼克号生还者分析')
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#导入数据
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
#print (data_train.info())

#对数据集进行预处理
data_train = data_train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#choose training data to predict age
age_df = data_train[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(data_train['Age'].notnull())]
age_df_isnull = age_df.loc[(data_train['Age'].isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
data_train.loc[data_train['Age'].isnull(), ['Age']]= predictAges
data_train['Embarked']=data_train['Embarked'].fillna('S')
#print (data_train.info())
X = data_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = data_train['Survived']
print (X)
