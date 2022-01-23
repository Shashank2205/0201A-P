# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 19:40:09 2022

@author: vikas
"""

#Logistic Regression


import pandas as pd

df = pd.read_csv('binary.csv')

x = df['gre'].values.reshape((-1,1))
y = df['admit'].values


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
ypred = model.predict(x)
ypred

plt.scatter(x,y)
plt.scatter(x,ypred)



from sklearn.linear_model import LogisticRegression
model  = LogisticRegression()
model.fit(x,y)
ypred = model.predict(x)
ypred

plt.scatter(x,y)
plt.scatter(x,ypred)




import math

ls= list(range(-6,6))

y=[]
for x in ls:
    y.append((1)/(1+math.exp(-x)))


import matplotlib.pyplot as plt

plt.scatter(ls,y)




#Case 1
import pandas as pd
df = pd.read_csv('titanic_all_numeric.csv')
df.columns

x = df.drop('survived',axis=1).values
x
y = df['survived'].values
y

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x,y)

pred = log.predict(x)
pred

df['pred'] = pred
df.to_csv('titanic.csv')

from sklearn.metrics import accuracy_score
accuracy_score(y, pred)



y = [0,1,1,0,1,0,1,1,1,0]
pred = [0,1,1,0,1,0,1,1,1,0]

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y,pred)
cf

TP =cf[0,0]
FN = cf[0,1]
FP =cf[1,0]
TN = cf[1,1]

accuracy = (TP+TN)/(TP+TN+FP+FN)

accuracy *100


y = [0,0,0,0,0,0,0,0,1,1]
pred = [0,0,0,0,0,0,0,0,0,0]

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y,pred)
cf

TP =cf[0,0]
FN = cf[0,1]
FP =cf[1,0]
TN = cf[1,1]

accuracy = (TP+TN)/(TP+TN+FP+FN)

accuracy *100

from sklearn.metrics import classification_report

print(classification_report(y, pred))





import pandas as pd
df = pd.read_csv('titanic_all_numeric.csv')
df.columns

x = df.drop('survived',axis=1).values
x
y = df['survived'].values
y

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x,y)

pred = log.predict(x)
pred

from sklearn.metrics import classification_report
print(classification_report(y, pred))





import pandas as pd
df = pd.read_csv('titanic_all_numeric.csv')
df.columns

x = df.drop('survived',axis=1).values
x
y = df['survived'].values
y

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(x,y, test_size=0.1)
import numpy as np
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


df.columns
xval = np.array([1,15,1,1,11.5,0,True,0,0,1]).reshape((1,-1))
xval.shape

log.predict(xval)


#case 

import pandas as pd
df = pd.read_csv('loan_data.csv')
df.columns

x = df.drop('not.fully.paid',axis=1).values
x
y = df['not.fully.paid'].values
y


df['not.fully.paid'].value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(x,y, test_size=0.3)
import numpy as np
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))



from imblearn.over_sampling import SMOTE
over = SMOTE(sampling_strategy=1)
xs, ys = over.fit_resample(x,y)

pd.Series(ys).value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(xs,ys, test_size=0.3)
import numpy as np
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))






#case 

import pandas as pd
df = pd.read_csv('smoker.csv')
df.columns

df


import numpy as np
np.sum(df.isna())

df = df.dropna()


x = df.drop('TenYearCHD',axis=1).values
x
y = df['TenYearCHD'].values
y


df['TenYearCHD'].value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(x,y, test_size=0.3)
import numpy as np
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))



from imblearn.over_sampling import SMOTE
over = SMOTE(sampling_strategy=1)
xs, ys = over.fit_resample(x,y)

pd.Series(ys).value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(xs,ys, test_size=0.3)
import numpy as np
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))





#Decision Tree Classifier


from sklearn import tree
import numpy as np

X = np.array([[10], [20], [15], [30], [18], [17]])
Y = np.array([0,1,0,1,1,0])  #class labels -  0- play no, 1- play yes
X
Y
X.shape
Y.shape

clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)

tree.plot_tree(decision_tree= clf)


#case

import pandas as pd
df = pd.read_csv('loan_data.csv')
df.columns

x = df.drop('not.fully.paid',axis=1).values
x
y = df['not.fully.paid'].values
y


df['not.fully.paid'].value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(x,y, test_size=0.3)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
log = DecisionTreeClassifier()
log.fit(xtrain,ytrain)
tree.plot_tree(decision_tree= log)




#Case
import pandas as pd
df = pd.read_csv('loan_data.csv')
df.columns

x = df.drop('not.fully.paid',axis=1).values
x
y = df['not.fully.paid'].values
y


df['not.fully.paid'].value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(x,y, test_size=0.3)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
log = DecisionTreeClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


from imblearn.over_sampling import SMOTE
over = SMOTE(sampling_strategy=1)
xs, ys = over.fit_resample(x,y)

pd.Series(ys).value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(xs,ys, test_size=0.3)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
log = DecisionTreeClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))




from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))




##Case
import pandas as pd
df = pd.read_csv('smoker.csv')
df.columns

df = df.dropna()

x = df.drop('TenYearCHD',axis=1).values
x
y = df['TenYearCHD'].values
y



from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(x,y, test_size=0.3)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
log = DecisionTreeClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


from imblearn.over_sampling import SMOTE
over = SMOTE(sampling_strategy=1)
xs, ys = over.fit_resample(x,y)

pd.Series(ys).value_counts()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(xs,ys, test_size=0.3)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
log = DecisionTreeClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))




from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))



from sklearn.ensemble import RandomForestClassifier
log = RandomForestClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


from sklearn.neighbors import KNeighborsClassifier
log = KNeighborsClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(log, xs, ys, cv=10)
scores

import numpy as np

np.mean(scores)


from sklearn.ensemble import ExtraTreesClassifier
log = ExtraTreesClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(log, xs, ys, cv=10)
scores

import numpy as np

np.mean(scores)



from sklearn.ensemble import HistGradientBoostingClassifier
log = HistGradientBoostingClassifier()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(log, xs, ys, cv=10)
scores

import numpy as np

np.mean(scores)



from sklearn.naive_bayes import GaussianNB
log = GaussianNB()
log.fit(xtrain,ytrain)

pred = log.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(log, xs, ys, cv=10)
scores

import numpy as np
np.mean(scores)




#Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {'x1': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'x2': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
  
df = pd.DataFrame(data,columns=['x1','x2'])
print (df)

plt.scatter(data['x1'], data['x2'])


from sklearn.cluster import KMeans

kmeans  = KMeans(n_clusters=2)
kmeans.fit(df)

kmeans.labels_

centroids = kmeans.cluster_centers_
centroids

plt.scatter(df['x1'], df['x2'], c=kmeans.labels_, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker="*")




from sklearn.cluster import KMeans

kmeans  = KMeans(n_clusters=2)
kmeans.fit(df)

kmeans.labels_

centroids = kmeans.cluster_centers_
centroids

plt.scatter(df['x1'], df['x2'], c=kmeans.labels_, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker="*")

kmeans.inertia_


sse=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)


sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();





from kneed import KneeLocator
k1 = KneeLocator(x = range(1,11), y=sse, curve='convex', direction='decreasing')

k1.elbow


from sklearn.cluster import KMeans

kmeans  = KMeans(n_clusters=3)
kmeans.fit(df)

kmeans.labels_

centroids = kmeans.cluster_centers_
centroids

plt.scatter(df['x1'], df['x2'], c=kmeans.labels_, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker="*")


#Case
import pandas as pd
df = pd.read_csv('FoodOrder.csv')
df1=df

df.columns

df = df.drop('Cust_Id',axis=1)
df.dtypes


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

df['NPS_Category'] = enc.fit_transform(df['NPS_Category'])

df1 = df.select_dtypes(include='object')

col = df1.columns

for c in col:
    enc = LabelEncoder()
    df[c] = enc.fit_transform(df[c]) 

df.dtypes



from sklearn.cluster import KMeans
sse=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)


sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();





from kneed import KneeLocator
k1 = KneeLocator(x = range(1,11), y=sse, curve='convex', direction='decreasing')

k1.elbow

kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
kmeans.labels_


df1 = pd.read_csv('FoodOrder.csv')

df1['label'] = kmeans.labels_

df1.to_csv('FoodLabels.csv')







#Case
import pandas as pd
df = pd.read_csv('mt.csv')

df.columns

df = df.drop('Unnamed: 0',axis=1)
df.dtypes



from sklearn.cluster import KMeans
sse=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)


sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();





from kneed import KneeLocator
k1 = KneeLocator(x = range(1,11), y=sse, curve='convex', direction='decreasing')

k1.elbow

kmeans = KMeans(n_clusters=2)
kmeans.fit(df)
kmeans.labels_


df1 = pd.read_csv('mt.csv')

df1['label'] = kmeans.labels_

df1.to_csv('MTLabels.csv')


























