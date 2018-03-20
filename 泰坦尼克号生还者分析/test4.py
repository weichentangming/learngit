import numpy as np
from numpy import newaxis
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
os.chdir('/home/weifeng/learngit/泰坦尼克号生还者分析')
pd.set_option('display.max_rows',None)
train_df = pd.read_csv("train.csv")
test_df =pd.read_csv("test.csv")
data = train_df.append(test_df)
data = data.drop(['Cabin','PassengerId','Ticket'],axis=1)
#sex
data['Sex'] = data['Sex'].map({'female':1,'male':0})
#data['Pclass*Sex'] = data['Pclass']*data['Sex']
#data['Pclass*Sex'] = pd.factorize(data['Pclass*Sex'])[0]
#print(data['Pclass*Sex'])
#name
data['Title'] = data['Name']
for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.',expand = True)
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss', 'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs','Dr':'Mr','Rev':'Mr'}
data.replace({'Title':mapping},inplace=True)
titles = ['Miss','Mr','Mrs',]
#data['Title'] = data['Title'].replace(['Capt', 'Col', 'Dona', 'Dr','Jonkheer','Major', 'Rev', 'Sir','Don'], 'Male_Rare')

#data['Title'] = data['Title'].replace(['Countess', 'Lady','Mlle', 'Mme', 'Ms'], 'Female_Rare')
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2}
data['Title'] = data['Title'].map(title_mapping)
enc = OneHotEncoder()
#print(data['Title'].T.shape)
#data['Title'] =np.vstack(data['Title'][:,newaxis])
#print(data['Title'].shape)
#enc.fit(data['Title'])
#data['Title_code'] = enc.transfrom(data['Title']).toarray()
#print(data['Title'].shape)
##families
data['Family_size'] = data['Parch'] + data['SibSp']
#data.loc[data['Family_size']==0,'Family'] = 0
#data.loc[(data['Family_size']>0&(data['Family_size']<4)),'Family'] = 1
#data.loc[data['Family_size']>4,'Family'] = 2
label = LabelEncoder()
data['Familybin'] = pd.cut(data['Family_size'],3)
data['Family_code'] = label.fit_transform(data['Familybin'])
#print(data['Family_code'])
##ages
age_to_impute = data['Age'].median()
data.loc[(data['Age'].isnull()),'Age'] = age_to_impute
#data.loc[data['Age']<=20,'Age'] = 4
#data.loc[data['Age']>60,'Age'] = 7
#data.loc[(data['Age']>20)&(data['Age']<=40),'Age'] = 5
#data.loc[(data['Age']>40)&(data['Age']<=60),'Age'] = 6
data['Agebin'] = pd.qcut(data['Age'],4)
#print(data['Agebin'])
data['Age_code'] = label.fit_transform(data['Agebin'])
#print(data['Age_code'])
#Emnarked
data['Embarked'] = data['Embarked'].fillna('S')
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
#Fare
data['Fare'].fillna(data['Fare'].dropna().median(), inplace=True)
#data.loc[data['Fare'] <= 128.082, 'Fare'] = 0
#data.loc[(data['Fare'] > 128.082) & (data['Fare'] <= 256.165), 'Fare'] = 1
#data.loc[(data['Fare'] > 256.165) & (data['Fare'] <= 384.247), 'Fare'] = 2
#data.loc[data['Fare'] > 384.247, 'Fare'] = 3
#data['Fare'] = data['Fare'].astype(int)
data['Farebin'] = pd.cut(data['Fare'],4)
#print(data['Farebin'])
data['Fare_code'] = label.fit_transform(data['Farebin'])
#print(data['Age_code'])
#取出要用的数据
data = data.drop(['Name','Parch','SibSp','Family_size','Age','Fare','Familybin','Agebin','Farebin'],axis=1)

train = data[:891]
test = data[891:]
test = test.drop(['Survived'],axis=1)
y = train['Survived']
X = train.drop(['Survived'],axis=1)
print(y.shape,X.shape)
#std_scaler = StandardScaler()
#X = std_scaler.fit_transform(X)
#test = std_scaler.transform(test)
enc.fit(X)
X = enc.transform(X).toarray()
enc.fit(test)
test = enc.transform(test).toarray()

#X = preprocessing.normalize(X,norm='l1')
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=6, p=2, weights='uniform')

#knn =BaggingClassifier(KNeighborsClassifier(),n_estimators=40,max_samples=0.5,max_features=0.5)
#knn = KNeighborsClassifier(n_neighbors=6)
gbc = GradientBoostingClassifier(n_estimators=80)
#交叉验证
scores = cross_validation.cross_val_score(gbc,X,y,cv=5)
gbc.fit(X,y)
gbc_predict = gbc.predict(test).astype(int)
passengers = np.arange(892,1310)
name1 = ['PassengerId']
name2 =['Survived']
columns1 = pd.DataFrame(columns = name1,data = passengers)
columns2 = pd.DataFrame(columns = name2,data = gbc_predict)
results = columns1.join(columns2)
results = results.set_index('PassengerId')
print(scores.mean())
results.to_csv('new1.csv')
