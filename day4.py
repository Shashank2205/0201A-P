# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:29:37 2022

@author: vikas
"""


from pydataset import data
data('')

mt = data('mtcars')
mt

type(mt)


import pandas as pd

mt.to_csv('mt.csv')


import pandas as pd

df = pd.read_csv('mt.csv')
df


df = pd.read_csv('20denco.csv')
df

type(df)




import pandas as pd

s1 = pd.Series([1,4,3,7,6])
s1

s2 = pd.Series([1,4,3,5,6,2], index=range(101,108))
s2

s3 = pd.Series([1,4,3,5,6], index=['a', 'b','b','c','a'])
s3

s3['a']
s3.loc['a']
s3.iloc[0]
s3.iloc[3]

s3


course = ['BBA', 'MBA', 'MTech', 'BTech', 'MBA']
strength =[100,50, 20, 80, 66]
fees = [1.2, 1.5, 1.3, 1.4, 1.5]

d1 = {'Course':course, 'Strength':strength, 'Fees':fees}
d1

import pandas as pd
df = pd.DataFrame(d1)
df

df.to_csv('stdinfo.csv')


df.index
df.columns
df.values

df.iloc[0:2]

df.count()

df

df['Course']

df['Course'] == 'MBA'

df[df['Course'] == 'MBA']


import pandas as pd

rno = range(1, 11)
rno

name=[]
for i in range(1,11):
    name.append('Student'+str(i))
name

gender = ['Male', 'Female']

import numpy as np
glist = np.random.choice(gender, size=10)
glist

course = np.random.choice(['MBA', 'BBA', 'BTech', 'MTech'], size=10)
course

Marks = np.random.randint(0, 101, size=10)
Marks


sinfo = pd.DataFrame({'Rno':rno,'Name':name,'Gender':glist,'Course':course,
                      'Marks':Marks})

sinfo



import pandas as pd

rno = range(1, 1000001)
rno

name=[]
for i in range(1,1000001):
    name.append('Student'+str(i))
name

gender = ['Male', 'Female']

import numpy as np
glist = np.random.choice(gender, size=1000000)
glist

course = np.random.choice(['MBA', 'BBA', 'BTech', 'MTech'], size=1000000)
course

Marks = np.random.randint(0, 101, size=1000000)
Marks


sinfo = pd.DataFrame({'Rno':rno,'Name':name,'Gender':glist,'Course':course,
                      'Marks':Marks})

sinfo



sinfo.to_csv('Data.csv')

sinfo[(sinfo['Marks']>95) & (sinfo['Course']=='BBA') ]

sinfo.describe()



rinfo = sinfo[sinfo['Marks']>98]

type(rinfo)


df = pd.read_csv('Data.csv')
df.count()

df.columns

sum(df['Name'].isnull())
sum(df['Name'].notnull())

df1 = df.dropna(axis=0)
df1.count()

df1 = df.dropna(axis=1)
df1


df2 = df.fillna(0)

df2.count()

df2.to_csv('Data1.csv')



df = pd.read_csv('airline.csv')

df

df.plot()

df1 = df.fillna(0)
df1.plot()


df2 = df.fillna(method='ffill')
df2.plot()

s1 = pd.Series([1,None ,3,4,3,1,None ,8,None ,None ,None ,9])

s1.fillna(method='ffill')

s1.fillna(method='bfill')


pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, None], [None, None, None, None, None], ['kanika', 28, None, 5000, None], ['tanvi', 20, 'F', None, None], ['poonam',45,'F',None,None],['upen',None,'M',None, None]])
pd4

pd4.dropna()

pd4.dropna(axis=1)

pd4.dropna(axis='rows')
pd4.dropna(axis='columns')

pd4
pd4.dropna(axis='rows',how='all')

pd4.dropna(axis='columns',how='all')

pd4.dropna(axis='rows',how='any')

pd4
pd4.dropna(axis='rows',thresh = 3)



grades1 = {'subject1': ['A1','B1','A2','A3'],'subject2': ['A2','A1','B2','B3']   }
grades1

df1 = pd.DataFrame(grades1)
df1

grades2 = {'subject3': ['A1','B1','A2','A3'],'subject4': ['A2','A1','B2','B3']}

df2 = pd.DataFrame(grades2)
df2

df1
df2
pd.concat([df1,df2])

pd.concat([df1,df2], axis=1)


grades3 = {'subject1': ['A1','B1','A2','A3'],'subject4': ['A2','A1','B2','B3']}

df3 = pd.DataFrame(grades3)

df1
df3

pd.concat([df1,df3])








import pandas as pd
#Join

rollno = pd.Series(range(1,11))

rollno

name = pd.Series(["student" + str(i) for i in range(1,11)])
name

genderlist  = ['M','F']

import random

gender = random.choices(genderlist, k=10)
gender

random.choices(population=genderlist,weights=[0.4, 0.6],k=10)

import numpy as np
#numpy.random.choice(items, trials, p=probs)
np.random.choice(a=genderlist, size=10, p=[.2,.8])


import numpy as np
marks1 = np.random.randint(40,100,size=10)
marks1


pd5 = pd.DataFrame({'rollno':rollno, 'name':name, 'gender':gender, 'marks1':marks1})

pd5

#course = random.choices( population=['BBA','MBA','BTECH'] ,weights=[0.4, 0.3,0.3],k=10)
course = np.random.choice(a=['BBA','MBA','BTECH'], size=10)
course

marks2 = np.random.randint(40,100,size=10)

marks2

pd6 = pd.DataFrame({'rollno':rollno, 'course':course, 'marks2':marks2})
pd6
pd5




fees = pd.DataFrame({'course':['BBA','MBA','BTECH', 'MTECH'], 'fees':[100000, 200000, 150000, 220000]})

fees


pd5
pd6
pd.merge(pd5, pd6)


rollno = pd.Series(range(6,16))

rollno

name = pd.Series(["student" + str(i) for i in range(6,16)])
name

pd7 = pd.DataFrame({'rollno':rollno, 'course':course, 'marks2':marks2})


pd5
pd7

pd.merge(pd5, pd7, how='inner')
pd.merge(pd5, pd7, how='outer')
pd.merge(pd5, pd7, how='left')
pd.merge(pd5, pd7, how='right')







import pandas as pd
import numpy as np

rollno = pd.Series(range(1,1001))

rollno

name = pd.Series(["student" + str(i) for i in range(1,1001)])
name

genderlist  = ['M','F']

import random
#gender = random.choices(genderlist, k=1000)
gender= np.random.choice(a=genderlist, size=1000,replace=True, p=[.6,.4])
gender


import collections

collections.Counter(gender)

marks1 = np.random.randint(40,100,size=1000)

marks2 = np.random.randint(40,100,size=1000)

fees = np.random.randint(50000,100000,size=1000)

fees.mean()

course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000, p=[0.4, 0.5,0.09,0.01])

course
collections.Counter(course)

city = np.random.choice(a=['Delhi', 'Gurugram','Noida','Faridabad'], size=1000, replace=True, p=[.4,.2,.2,.2])

collections.Counter(city)


course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000, p=[0.4, 0.5,0.09,0.01])
pd8 = pd.DataFrame({'rollno':rollno, 'name':name, 'course':course, 'gender':gender, 'marks1':marks1,'marks2':marks2, 'fees':fees,'city':city})
pd8


pd8.groupby('course')

pd8.groupby('course').size()
pd8.groupby(['course','gender']).size()
pd8.groupby(['gender','course']).size()


pd8.columns

pd8.groupby(['course', 'gender','city']).count()
pd8.groupby(['course', 'gender','city']).size()

pd8.groupby(['course', 'gender','city']).agg({'marks1':"mean"})

pd8

pd8.groupby(['course', 'gender','city']).aggregate(
    {'fees':["sum", "mean", 'max','min']})



pd9 = pd8.groupby(['course', 'gender','city']).aggregate(
    {'fees':["sum", "mean"], 'marks1':['mean', 'min']})

pd9.to_csv('data.csv')





import pandas as pd

df = pd.read_csv('20denco.csv')
df.columns

df.groupby('custname').size()
df.groupby('custname').count()
df.groupby('custname').size().sort_values(ascending=False)
df.groupby('custname').size().sort_values(ascending=False).head(10)

df.groupby('custname').agg({'revenue':'sum'}).sort_values(ascending=False, by='revenue')
df.groupby('custname').agg({'revenue':'sum'}).sort_values(ascending=False, by='revenue').head(10)


df.columns

prevenue=df.groupby('partnum').agg({'revenue':'sum'}).sort_values(ascending=False, by='revenue').head(10)


marg = df.groupby('partnum').agg({'margin':'sum'}).sort_values(ascending=False, by='margin').head(10)



'''
•	Who are the most loyal Customers -
o	Make customer table, See customer transaction,
o	Sort Customer Transaction,
o	How many times are these customers buying from me
o	Select the Top 5 or 10 rows (Sorted in Descending Order of Frequency)
•	Which customers contribute the most to their revenue
o	Sum the revenue by each customer
o	Sort revenue by customers in descending Order
•	What part numbers bring in to significant portion of revenue
o	Sum/ Group the revenue by part no
o	Sort the revenue by decreasing order
o	Top revenue by part nos
•	What parts have the highest profit margin ?
o	Sum the margin by partno
o	Sort the margin by decreasing order
o	Parts contributing highest profit margin
•	Who are their top buying customers
•	Who are the customers who are bringing more revenue

'''


df = pd.read_excel('20denco.xlsx')
df

df.to_excel('Data.xlsx')


#write to more than one sheet in the workbook, it is necessary to specify an ExcelWriter object:
with pd.ExcelWriter('Data.xlsx') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    marg.to_excel(writer, sheet_name='Margin')
    prevenue.to_excel(writer, sheet_name='Revenue')
    

df = pd.read_excel('Data.xlsx', sheet_name='Revenue')
df




import matplotlib.pyplot as plt
























































