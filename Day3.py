# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 19:27:21 2022

@author: vikas
"""

#Functions

print("Hi")


def table(j):
    for i in range(1,11):
        print("{0} * {1} = {2}".format(j,i,i*j))
    
table(2)

table(55)



#Declare, Define and Call

def f1():
    print("Hello")
       
f1()

f1()


def f2(name):
    print("Hello "+ name)


f2("Vikas")

f2() #TypeError: f2() missing 1 required positional argument: 'name'


f2('ABC', '2') #TypeError: f2() takes 1 positional argument but 2 were given


def f3(name="None"):
    print("Hello "+name)


f3('Vikas')

f3()


a= 10
b=20
print(a+b)
print(a-b)
print(a*b)
print(a**b)
print(a/b)





def f4():
    a = 10
    b = 20
    print(a+b)
    print(a-b)
    print(a*b)
    print(a**b)
    print(a/b)


f4()

f4()



def f5(a,b):
    print(a+b)
    print(a-b)
    print(a*b)
    print(a**b)
    print(a/b)



f5(10,30)

f5(110, 66)

f5(2,8)


print(f5(3,4))


l1 = [4,3,6,7,3,2]

print(max(l1))

m = max(l1)

print(m)



l1


def MAX(seq):
    m = 0
    for i in seq:
        if (i>m):
            m=i
    return(m)


a = MAX(l1)

print(a)




rno=[]
name=[]
course=[]


def Data(R, N, C):
    R.append(input("Enter Roll No-> "))
    N.append(input("Enter Name->    "))
    C.append(input("Enter Course->  "))
    return(R,N,C)


rno, name, course = Data(rno, name, course)


rno, name, course = Data(rno, name, course)


rno, name, course = Data(rno, name, course)




#Lambda

def f(x):
    return(x**2)


f(4)

f(8)


fl = lambda x: x**2


fl(66)

fl(88)


fl1 = lambda x,y: x**y

fl1(4,5)

fl1(2) #TypeError: <lambda>() missing 1 required positional argument: 'y'

fl1(22,3)


l1 = [5,4,7,6,2,1,9,8]

def sqq(i):
    return(i**2)

rl = []
for i in l1:
    rl.append(sqq(i))
rl

sq = lambda x: x**2
rl1 = list(map(sq, l1))


l2 = ['abc', 'bdd', 'sde']

l2[0] = l2[0].upper()

up = lambda x: x.upper()

rl2 = list(map(up,l2))
rl2


rl2 = list(map(lambda x: x.upper(),l2))
rl2


l1 = [5,4,7,6,2,1,9,8]

od = lambda x: x%2==0

list(map(od,l1))

list(filter(od, l1))



l2 = [-5,-4,-7,6,2,-1,9,-8]

od = lambda x: x >= 0

list(filter(od, l2))


l3 = ['4','5','3', 's', '4','d','4','6','q']

od = lambda x: x.isdigit()
list(filter(od, l3))



import random as rd

rd.randint(0,100000)

rd.randrange(0,1000,100)


l1  = [111,222,333,444,555]

rd.choice(l1)

lg = ['Male', 'Female']
rd.choice(lg)

rd.choices(lg, k=10)


import numpy as np

np.random.randint(1,10)

#1 Dim

n1 = np.random.randint(1,10, size=5)
n1

n1.shape


n2 = np.random.randint(1, 10, size=(3,2))
n2

sh = n2.shape

sh
sh[0]
sh[1]

n3 = np.random.randint(1,10, size= (3,4,3))
n3

n4 = np.random.randint(1,10, size= (2,3,4,5))
n4


n5 = np.random.randint(1,10, size=5)

n5

n5[ : ]
n5[0:5]

n5[1:5]

n5[1:4]

n5[:4]
n5[3:]

n5[-2:]

n5[-3]

n5[-4:-2]
n5


import numpy as np
n6 = np.random.randint(1,10, size=(6,4))
n6

n6.shape

n6[0]
n6[0,0]
n6[0,0:2]
n6[0,-2]
n6[0,:-2]


n6[1]
n6[1,0]
n6[1,0:2]
n6[1,-2]
n6[1,:-2]

n6[0:2]
n6[0:2,0]
n6[0:2, 0:2]

n6
n6[0:2]
n6[4:,0:2]
n6[2:4,2: ]


n6
n6[-1,-1]


n7 = np.arange(10)
n7

n7 = np.arange(3,10)
n7

n7 = np.arange(10,500, 10)
n7

n7.shape

n8 = n7.reshape((7,7))
n8

a = np.zeros((3,4))
a

type(a[0][0])

a = np.zeros((3,4)).astype('int')
a

b = a.astype('float')
b

b = a.astype('str')
b

c = np.ones((4,5))
c


np.eye(3,3)


np.linspace(0, 10, num=5)



central tedencies
mean
median
mode


l1= [5,6,3,2]

np.std(l1)
np.mean(l1)


n1 = np.random.randint(1, 10000, size=1000)

np.std(n1)
np.mean(n1)


from math import pow
pow((pow((5-4),2) + pow((6-4),2) + pow((3-4),2) + pow((2-4),2))/4, (1/2))

np.std(l1)



l1.sort()
l1

np.median(l1)

np.mode(l1)


Deviations
Standard Deviation
Varriance



import pandas as pd
df = pd.read_csv('20denco.csv')
df.columns
n3 = df['margin'].values



n3.shape

l1 = list(n3)

np.mean(l1)
np.median(l1)
np.std(l1)


np.floor([1.2, 1.6])
np.ceil([1.2, 1.6])
np.trunc([1.2, 1.6])
np.round([1.2, 1.6])

np.floor([-1.2, -1.6])
np.ceil([-1.2, -1.6])
np.trunc([-1.2, -1.6])
np.round([-1.2, -1.6])



np.round([1.223233,3.4432133],2)
np.round([1.223233,3.4432133],1)


n1 = np.random.randint(1,10, size = (3,5))
n1

n2 = np.random.randint(1,10, size = (4,5))
n2

n3 = np.concatenate([n1,n2], axis=0)
n3



n1 = np.random.randint(1,10, size = (3,5))
n1

n2 = np.random.randint(1,10, size = (3,6))
n2

n3 = np.concatenate([n1,n2], axis=1)
n3



n4 = np.random.randint(1, 10, size=(10000,5))
n4

n4.shape

n5 = np.split(n4, [8000,])

type(n5)
len(n5)
n5[0]


n5[0].shape
n5[1].shape


n6 = np.random.randint(1,10, size=(9,4))
n6
n6.mean()
np.mean(n6)
n6.mean(axis=0)
n6.mean(axis=1)

n6.sum()
n6.sum(axis=0)
n6.sum(axis=1)


n6 > 6
n6
np.sum(n6>6)



















