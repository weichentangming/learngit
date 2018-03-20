#coding=utf-8
import os
os.chdir('/home/weifeng/learngit/泰坦尼克号生还者分析')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas import Series,DataFrame
from matplotlib.font_manager import *
#matplotlib.rcParams['font.sans-serif']=['ukai ']
myfont = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
data =pd.read_csv('train.csv')
#print (data.head(10))
#print (data.isnull().sum())  
fig = plt.figure(figsize=(10,10))
fig.set(alpha=0.5)
plt.subplot(231)
data.Survived.value_counts().plot(kind='bar') 
plt.title('获救情况',fontproperties=myfont)
plt.ylabel('人数',fontproperties=myfont)

plt.subplot(232)
data.Pclass.value_counts().plot(kind='bar')
plt.title('乘客等级',fontproperties=myfont)


plt.subplot(233)
plt.scatter(data.Survived,data.Age)

plt.subplot2grid((2,3),(1,0), colspan=2)
data.Age[data.Pclass == 1].plot(kind='kde')   
data.Age[data.Pclass == 2].plot(kind='kde')
data.Age[data.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄",fontproperties=myfont)# plots an axis lable
plt.ylabel(u"密度",fontproperties=myfont) 
plt.title(u"各等级的乘客年龄分布",fontproperties=myfont)
plt.legend((u'头等舱', u'2等舱',u'3等舱'),prop = myfont,loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数",fontproperties=myfont) 
plt.show()

#fig2 = plt.figure()
plt.subplot(231)
fig.set(alpha=0.2) #设定图表颜色参数
Survived_0 = data.Pclass[data.Survived ==0].value_counts()
Survived_1 = data.Pclass[data.Survived ==1].value_counts()
#df = pd.DataFrame({u'获救'：Survived_1,u'未获救'：Survived_1})
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar',stacked=True)

plt.subplot(232)
Survived_m = data.Survived[data.Sex == 'male'].value_counts()
Survived_f = data.Survived[data.Sex == 'female'].value_counts()
df1=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df1.plot(kind='bar', stacked=True)
plt.show()

