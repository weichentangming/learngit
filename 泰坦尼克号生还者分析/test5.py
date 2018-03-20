import numpy as np
import pandas as pd
import os
from sklearn import cross_validation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
os.chdir('/home/weifeng/learngit/泰坦尼克号生还者分析')
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_rows',None)

train_df = pd.read_csv("train.csv")
test_df =pd.read_csv("test.csv")
data_df = train_df.append(test_df)

data_df['Title'] = data_df['Name']
# Cleaning name and extracting Title
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
#print(data_df['Title'])
# Replacing rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss','Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
#print(data_df['Age'].groupby(df['Title'])
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
                                  
# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]
#print(data_df.info())
# Dropping Title feature
data_df.drop('Title', axis = 1, inplace = True)
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
#print(data_df['Last_Name'])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId','SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
        if (len(grp_df) != 1):
                for ind, row in grp_df.iterrows():
                        smax = grp_df.drop(ind)['Survived'].max()
                        smin = grp_df.drop(ind)['Survived'].min()
                        passID = row['PassengerId']
                        if (smax == 1.0):
                                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                        elif (smin==0.0):
                                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
for _, grp_df in data_df.groupby('Ticket'):
        if (len(grp_df) != 1):
                for ind, row in grp_df.iterrows():
                        if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                                smax = grp_df.drop(ind)['Survived'].max()
                                smin = grp_df.drop(ind)['Survived'].min()
                                passID = row['PassengerId']
                                if (smax == 1.0):
                                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                                elif (smin==0.0):
                                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]
data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

# Making Bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)
#print(data_df['FareBin'])
label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])
train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]
train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)
data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)
#print(data_df['AgeBin'])
label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])
#print(data_df['AgeBin_Code'])
train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]
train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin','Embarked'], axis = 1, inplace = True)

#训练
X = train_df.drop('Survived', 1)
y = train_df['Survived']
X_test = test_df.copy()
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=6, p=2, weights='uniform')
#knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
y_predict = knn.predict(X_test)
gbc = GradientBoostingClassifier(n_estimators=100)
#交叉验证
scores = cross_validation.cross_val_score(knn,X,y,cv=5)
gbc.fit(X,y)
gbc_predict = gbc.predict(X_test).astype(int)
print(scores.mean())
passengers = np.arange(892,1310)
name1 = ['PassengerId']
name2 =['Survived']
columns1 = pd.DataFrame(columns = name1,data = passengers)
columns2 = pd.DataFrame(columns = name2,data = y_predict)
results = columns1.join(columns2)
results = results.set_index('PassengerId')
#results.to_csv('new3.csv')
