# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 20:27:54 2022

@author: vikas
"""

print()
print("Hi Welocme to Python")
print("Welcome To Analytics")

print("Hi Welcome to Python", "Welcome to Analytics")
print("Hi Welcome to Python", "Welcome to Analytics", sep="-")



print("Hi Welcome to Python", "Welcome to Analytics", end="-----")
print("Hi Welcome to Python", "Welcome to Analytics", sep="-")


print?
help(print)



!pip install pandas


import pandas

pandas.__version__

import matplotlib as plt

plt.__version__


import pandas as pd

pd.__version__


s = "HI"

print(s)

print("s")

country= "India"
print("I live in ", country, sep='--')


print("I live in country")


country= "UAE"
print("I live in ", country)


print("I live in {0}".format(country))

liveC ='India'
workC ='UAE'

print("I live in {0} and I work in {1}".format(liveC,workC))

print("I live in {1} and I work in {0}".format(liveC,workC))

print("I live in {1} and I work in {1}".format(liveC,workC))

liveC ="USA"
print("I live in {0} and I work in {1}".format(workC,liveC))


liveC = input()

liveC = input("Enter Live Country->")
workC = input("Enter Work Country->")
print("I live in {0} and I work in {1}".format(workC,liveC))


a = 10
b = 10.4


c = input("Enter a Value")

d = int(c)

a + b
a + c
a + d
#Trype Conversion or Type Casting

c = int(input("Enter a Value"))


s = "Marks are "
m = 60

sm = s + str(m)
sm

"Marks are 60"


i = 3
i_f = float(i)
i_f

i_s = str(i)
i_s

f = 4.55
f_i = int(f)
f_i

f_s = str(f)
f_s

s = "33"
s_i= int(s)
s_i

s = "332.2"
s_f = float(s)
s_f




s = "?"
ord(s)

t1 = "Hello : Hi How = are @ > You"
t2 = ""

for i in t1:
    if ((ord(i)>=65 and ord(i)<=90) or (ord(i)>=97 and ord(i)<=122) or ord(i)==32):
        t2 = t2+i

print(t2)




#Operators

x = 5

print(x)

print(type(x))

print(x+1)
print(x-1)
print(x*2)

print(x/2)
print(x%2)


ch = int(input("Enter a Number->"))
if(ch%2 == 0):
    print("Even Number")
else:
    print ("Odd Number")


print(x**2)
print(x**3)

print(x**(1/2))


x

x = x+5

x += 24

y = x +33


#Boolean

t = True
f = False

print(type(t))


a = 30
b = 40

a>b
a<b
a<=b
a>=b
a==b
a!=b


'''
AND

A B O
0 0 0
0 1 0
1 0 0
1 1 1

OR
A B O
0 0 0
0 1 1
1 0 1
1 1 1

Not
A O
0 1
1 0
'''

a=10
b=20
c=30

a>b and b>c
a>b and b<c
a<b and b>c
a<b and b<c


a>b or b>c
a>b or b<c
a<b or b>c
a<b or b<c


not (a>b)
not (a<b)

#String Handling

h = "hello"
w = 'python'

hw = h + w

hw = h +" "+ w

print(hw)

hw = hw.capitalize()
print(hw)

hw = hw.upper()
print(hw)

hw = hw.lower()
print(hw)

hw = "I like JAVA work on JAVA"
hw = hw.replace("JAVA", 'Python')
print(hw)


hw  = "I -- Like to work on Python"
hw = hw.replace('--','')
print(hw)

hw = 'Python'
print(hw)
hw = hw.ljust(12)
print(hw)

hw = 'Python'
print(hw)
hw = hw.rjust(12)
print(hw)

hw = 'Python'
print(hw)
hw = hw.center(12)
print(hw)


name = input("Enter your Name->")
print(name)

name = name.strip()
print(name)


name = "Vikas Khullar"

n = name.split(" ")

n[0]
n[1]

