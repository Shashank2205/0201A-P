# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 20:55:09 2022

@author: vikas
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([5,15,25,35,45,55]).reshape((-1,1))
x.shape

y = np.array([5,20,14,32,22,38])
y.shape

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

model.intercept_
model.coef_

ypred = model.predict(x)
ypred


plt.scatter(x,y)
plt.plot(x,ypred)
plt.scatter(x,ypred)




import pandas as pd

df = pd.read_csv('datahouse.csv')

df.columns

x = df['area'].values.reshape((-1,1))
x.shape

y = df['price'].values
y.shape

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

ypred = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,ypred)

r2 = model.score(x,y)
print(r2)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Housing.csv')
df.columns
df
df = df.dropna(axis=0)
df


x = df['area'].values.reshape((-1,1))
x.shape
y = df['price'].values



from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)
r2 = model.score(x,y)
r2

ypred = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,ypred)

0.9*1000000
y
plt.boxplot(y)






df = pd.read_csv('Housing.csv')
df = df.dropna(axis=0)

max(df['price'])

df = df[df['price']<=9000000]
df = df[df['area']<=10000]

df
df

x = df['area'].values.reshape((-1,1))
x.shape
y = df['price'].values



from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)
r2 = model.score(x,y)
r2

ypred = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,ypred)

plt.boxplot(y)
plt.boxplot(x)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('age.csv')

df.columns
df.dtypes

x = df.drop('Age', axis=1).values
x.shape

y = df['Age'].values
y.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)

r2 = model.score(x,y)
r2

import seaborn as sns
pp = sns.pairplot(df, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('USA_Housing.csv')

df.columns
df.dtypes

x = df.drop('Price', axis=1).values
x.shape

y = df['Price'].values
y.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)

r2 = model.score(x,y)
r2

import seaborn as sns
pp = sns.pairplot(df, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))



income=int(input("Enter Income"))
age=int(input("Enter Age"))
room=int(input("Enter room"))
bedroom=int(input("Enter Bedroom"))
population=int(input("Enter Population"))

data = np.array([income,age,room, bedroom, population]).reshape((1,-1))
data.shape

data
price = model.predict(data)

print('Predicted Price is-> ',price)






import statsmodels.api as sm
import matplotlib.pyplot as plt
from pydataset import data

mtcars = data('mtcars')
mtcars.columns

df1 = mtcars
df1.head(5)

from statsmodels.formula.api import ols

MTmodel1 = ols("mpg ~  wt + hp +cyl +disp+drat+qsec+vs+am+gear+carb", data=mtcars).fit()
print(MTmodel1.summary())


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(MTmodel1, fig=fig)


#Logistic Regression

import numpy as np
x = np.random.randint(1, 100, size=100 ).reshape((-1,1))
x.shape
y = np.random.randint(0,2, size=100)
y.shape


x
y

import matplotlib.pyplot as plt
plt.scatter(x,y)


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
ypred = model.predict(x)
ypred


plt.scatter(x,y)
plt.scatter(x,ypred)

model.score(x,y)



'''
from sklearn.linear_model import LogisticRegression
model  = LogisticRegression()
model.fit(x,y)
ypred = model.predict(x)
ypred

yprob = model.predict_proba(x)[:,1]
yprob
plt.scatter(x,y)
plt.scatter(x,yprob)
''










import math

math.exp(10)

ls= list(range(-6,6))

y=[]
for x in ls:
    y.append((1)/(1+math.exp(-x)))


import matplotlib.pyplot as plt

plt.scatter(ls,y)



















































