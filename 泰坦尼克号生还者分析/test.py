import os
os.chdir('/home/weifeng/learngit/泰坦尼克号生还者分析')
import xgboost as xgb
import pandas as pd
from mlxtend.classifier import StackingClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfTransformer
#导入数据
data_train =pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
#对数据进行预处理
data_train = data_train.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
data_test = data_test.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
data_train['Embarked']=data_train['Embarked'].fillna('S')
data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].mean())   
data_test['Embarked']=data_test['Embarked'].fillna('S')
age_df = data_train[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df1 = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(data_train['Age'].notnull())] 
age_df1_notnull = age_df1.loc[(data_test['Age'].notnull())]
age_df_isnull = age_df.loc[(data_train['Age'].isnull())]   
age_df1_isnull = age_df1.loc[(data_test['Age'].isnull())] 
X = age_df_notnull.values[:,1:] 
x = age_df1_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]                             
y = age_df1_notnull.values[:,0]
# use RandomForestRegression to train data                 
RFR = RandomForestRegressor(n_estimators=80, n_jobs=-1)                   
RFR1 = RandomForestRegressor(n_estimators=80, n_jobs=-1) 
RFR.fit(X,Y)                                               
RFR1.fit(x,y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])      
predictAges1 = RFR1.predict(age_df1_isnull.values[:,1:])
data_train.loc[data_train['Age'].isnull(), ['Age']]= predictAges
data_test.loc[data_test['Age'].isnull(), ['Age']]= predictAges1
#print (data_test.info())
#data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())
#data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())  
#data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].mean()) 
X = data_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = data_train['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
#print (X_train)


#对数据进行特征转换
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='recode'))
X = vec.fit_transform(X.to_dict(orient='recode'))
#X = preprocessing.normalize(X,norm='l1')
#X = preprocessing.scale(X)
#X = vec.fit_transform(X.to_dict(orient='recode'))
data_train = vec.fit_transform(data_train.to_dict(orient='recode'))
X_test = vec.fit_transform(X_test.to_dict(orient='recode'))
#transformer = TfidfTransformer(smooth_idf=False)
#X_train = transformer.fit_transform(X_train).toarray()
#print(X_train[0])

data_test = vec.fit_transform(data_test.to_dict(orient='recode'))
#print (data_test.shape,X_train.shape)
#使用单一决策树
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_predict = dtc.predict(data_test)

#使用随机森林
rfc = RandomForestClassifier(n_estimators=80)
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(data_test)

#使用梯度提升决策树
#param_range = np.int_([20,50,80,110,150])
#data_test.info()
gbc = GradientBoostingClassifier(n_estimators=80)
#train_loss,test_loss = validation_curve(gbc,X,y,param_name='n_estimators',param_range = param_range,cv=10,scoring='mean_squared_error')
#train_loss_mean = -np.mean(train_loss,axis=1)
#test_loss_mean = -np.mean(test_loss,axis=1)
gbc.fit(X_train,y_train)
gbc_predict=gbc.predict(data_test)
#scores = cross_val_score(gbc,X,y,cv=10,scoring='accuracy')

#使用xgboost
gbm = xgb.XGBClassifier(n_estimators=40)
gbm.fit(X_train,y_train)
xgb_predict = gbm.predict(data_test)

#使用KNN
knn =BaggingClassifier(KNeighborsClassifier(),n_estimators=40,max_samples=0.5,max_features=0.5)

knn.fit(X_train,y_train)

knn_predict = knn.predict(data_test)

#使用stack集成 随机森林，梯度提升决策树，KNN
clf1 = rfc
clf2 = gbc
clf3 = knn
lr = LogisticRegression()
slf = StackingClassifier(classifiers = [clf1,clf2,clf3],meta_classifier=lr)

slf.fit(X_train,y_train)

#交叉验证
scores = cross_validation.cross_val_score(gbm,X,y,cv=5)
#得分
dtc_score = dtc.score(X_test,y_test)
rfc_score = rfc.score(X_test,y_test)
#gbc_score = cross_val_score(gbc,X,y,cv=10)
gbc_score = gbc.score(X_test,y_test)
gbm_score = gbm.score(X_test,y_test)

knn_score = knn.score(X_test,y_test)
slf_score = slf.score(X_test,y_test)
#输出结果
print(np.mean(scores))
#print (dtc_score)
print(rfc_score)
print(gbc_score)
print(gbm_score)
#result1 = (418-np.sum(abs(gbc_predict-data_gender)))/418
#result2 = (418-np.sum(abs(xgb_predict-data_gender)))/418
#print(result1)
#print(result2)
#prin0t(gbc_predict.shape)
#print(data_gender.shape)
#data_train.info()
#print(data_gender.shape)
#可视化图形
#plt.plot(param_range,train_loss_mean,'o-',color="r",label="Training")
#plt.plot(param_range,test_loss_mean,'o-',color="g",label="Cross-validation")
#plt.show()

#_test.info()) 生成csv文件
#print(gbc_predict)
passengers = np.arange(892,1310)
#print(passengers.shape)
#result = np.concatenate((passengers,gbc_predict.reshape(418,1)),axis=0)
name1 = ['PassengerId']
name2 =['Survived']
columns1 = pd.DataFrame(columns = name1,data = passengers)
columns2 = pd.DataFrame(columns = name2,data = xgb_predict)
results = columns1.join(columns2)
results = results.set_index('PassengerId')
#results = results.drop(results.columns[0],axis=1)
#result3 = (418-np.sum(abs(results['Survived']-data_gender)))/418
#print(results)
results.to_csv('result2.csv')

