# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 19:18:31 2022

@author: vikas
"""



#List, Set, Dictionary, Tuple

"""
Hetrogeneous or Homogeneous
Ordered or Unordered
Mutable or Not Mutable
Indexed or Non-Indexed
""" 

#List

l1 = []

l2 = [1,2,3,4]

# Hetrogeneous

l3 = [10, 4.3, 'Python', True]


#Unordered
l4 = [4,3,7,6,9]
print(l4)


print(l2)

#Indexed

print(l3)

l3[0]
l3[1]
l3[2]
l3[3]
l3[4]



#Mutable or Changable

l4
l4[0] = 55
l4

l3
l3[2] = 'Java'



# Range
r1 = range(10)
r1

lr1 = list(r1)


print(lr1)


r2 = range(5, 21)
r2

lr2 = list(r2)
print(lr2)


r3 = range(10, 1001, 5)
r3

lr3 = list(r3)
print(lr3)


lr3[0]
lr3[1]
lr3[2]
lr3[3]




for i in lr3:
    print(i)
   


l4 = [4,3,7,6,9, 3,5, 3,4,2,1,6]

l4.count(4)


l5 = ['abc', 'xyz', 'abc', 'lmn']

l5.count('xyz')

l4

l4.append(12)
l4

print(l4.remove(7))
print(l4)


print(l4.remove(3))
print(l4)

print(l4.pop())
print(l4)

print(l4.pop(2))
print(l4)

l4.clear()

del l5



l4 = [4,3,7,6,9, 3,5, 3,4,2,1,6]

l4.sort()
print(l4)

l4.reverse()
print(l4)


l5 = ['oranges', 'mangoes', 'cherries']

l5.sort()
l5

l5.append('apples')
l5

l5.insert(1, 'bananas')

l5

l5.sort()
l5
l5.reverse()
l5
l5.sort()
l5


#Set
"""
Hetrogeneous or Homogeneous
Ordered or Unordered
Mutable or Not Mutable
Indexed or Non-Indexed
""" 

#Order
#Uniqueness
#Non Indexed
#Mutable

#Union
#Intersection
#Difference



s1 = {1}

s2 = {4,3,5,3,1,5,6,2}

#No Duplicacy and Uniqueness

#Ordered
s3 = {3,2,5,1}

#Non Indexed

s3[1] #TypeError: 'set' object is not subscriptable

s3.add(3)

s3.add(30)


for i in s3:
    print(i)


#Hetrogeneous

s4 = {'python', 111, 2.432, True}
s4

s4.add('Analytics')

s4.remove(True)
s4.remove(True) #KeyError: True

s4.discard(111)
s4.discard(111)


s4.pop()



teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}


teamA
teamB


teamA.union(teamB)
teamA.intersection(teamB)
teamA.difference(teamB)





#Dictionary

"""
Hetrogeneous or Homogeneous
Ordered or Unordered
Mutable or Not Mutable
Indexed or Non-Indexed
""" 


#Not Indexed
#Key Value

d1 = {}

sinfo = {'rno':1, 'name':'ABC', 'class': 'MTech'}

car = {'brand':'Toyota', 'name':'Innova', 'Color':'Black'}

car

car.keys()
car.items()
car.values()


#Mutable
car['year'] = 2021
car['name']

#Key:valued Pair not Indexed

car[1]

#Not ordered

#Hetrogeneous
car[0]='Nothing'
car


car.popitem()
car.pop('name')

print(car['brand'])
car.get('brand')



r1 = list(range(1,11))
r1

stinfo = {'rollno': r1 }

print(stinfo)


m1 = list(range(100, 110))

m1

stinfo['marks'] = m1

print(stinfo)

stinfo['rollno']

type(stinfo)

type(stinfo['rollno'])

type(stinfo['rollno'][0])

stinfo['rollno'][0]



{'rollno': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'marks': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}


#Tuple

a = 1,2

type(a)

t1 = ()
t1 = (2,3)

# Indexed

t2 = (2,4,1,3,7,6,5,9)

t2[1]
t2[7]

#Not Mutable or non  changeable

t2[1] = 3 # TypeError: 'tuple' object does not support item assignment

t2.count(2)



# Conditional Statements

a= 10
b=20

a=b

a= 10
b=20
c =10

a == b

a==c

a<b
a<=b

a>b
a>=b

a>=c

a!=b
a!=c



'''
if ('conditionl check')
{
   statement 1
   statement 2
   statement N 
}
'''



if (a<b):
    print("A is less than B")
    print("A is less")
    print("B is Greater")
print("B is Greater")


if (a>b):
    print("A is greater than B")


if (False):
    print("A is less than B")


a=20
b=10
if (a<b):
    print("A is less than B")
else:
    print("B is grater than A")




marks = int(input('Enter Marks'))

if(marks>90):
    print('O')
elif (marks>80 and marks<=90):
    print('A')
elif (marks>70 and marks<=80):
    print('B')
elif (marks>60 and marks<=70):
    print('C')
elif (marks>50 and marks<=60):
    print('D')
else:
    print('Fail')



#Iterative or Looping Statements


for i in range(10):
    print('Hello')


for i in range(10):
    print(i)


l1 = ['Abc', 'Zxx', 'Fmn', 'Dwww', 'Wqqq']

for i in l1:
    print(i)


for i in range(10):
    print('Hello'+ str(i))


name=[]
name.append('Student1')
name.append('Student2')

name=[]
for i in range(10):
    name.append('Student'+str(i))
    

print(name)


rollno = list(range(10))

d1 = {'rollno':rollno, 'name':name}
d1


len(name)


print("2 * 1 = 2")


for i in range(1,11):
    print("2 * {0} = {1}".format(i,i*2))
    
    
    print("2 * " , i , "=" , 2*i)
    
    print("2 * 1 = 2")

    
    print("2 * {0} = {1}".format(i,2*i))





for i in range(1,11):
    print("2 * {0} = {1}".format(i,i*2))
    


for j in range(1,6):
    for i in range(1,11):
        print("{0} * {1} = {2}".format(j,i,i*j))




j=2
for i in range(1,11):
    print("{0} * {1} = {2}".format(j,i,i*j))


for j in range(1, 6):
    for i in range(1,11):
        print("{0} * {1} = {2}".format(j,i,i*j))





while(False):
    print("Hi")




cnt=1

while(cnt<=10):
    print(cnt)
    cnt = cnt +1

i = 1

while(i<=10):
    print("2 * {0} = {1}".format(i, i*2))
    i=i+1



j=1
while(j<=3):
    i=1
    while(i<=10):
        print("{0} * {1} = {2}".format(j, i, i*j))
        i=i+1
    j=j+1




name =[]
chk=1

while(chk == 1):
    name.append(input('Enter Name->'))
    chk = int(input('Enter 1 to continue'))



chk=1
while(chk == 1):
    marks = int(input('Enter Marks'))
    if(marks>90):
        print('O')
    elif (marks>80 and marks<=90):
        print('A')
    elif (marks>70 and marks<=80):
        print('B')
    elif (marks>60 and marks<=70):
        print('C')
    elif (marks>50 and marks<=60):
        print('D')
    else:
        print('Fail')
    chk = int(input('Enter 1 to continue'))




team = ['India', 'Australia','Bangladesh','Nepal', 'England']   # 4elements   list index 0-3


for i in team:
    print (i)


team
for i in team:
    if(i =='Nepal'):
        print("In ",i)
    print ("Out",i)
print('Exited')


for i in team:
    if(i =='Nepal'):
        print("In ",i)
        break
    print ("Out",i)
print('Exited')


for i in team:
    if(i =='Nepal'):
        print("In ",i)
        continue
    print ("Out",i)
print('Exited')



for i in range(1,11):
    break


import numpy as np

rl = np.random.randint(0, 1000, size=1000)
cnt = 1
for i in rl:
    if(i == 9):
        print('found')
        print('Found at ', cnt)
        break
    cnt=cnt+1
print(cnt)
    




while(True):
    i = input("Enter Char->  ")
    if (ord(i)>=65 and ord(i)<=90):
        print ("Enterd",i, "   Exiting")
        break
    else:
        print ("Enterd",i, "   Again enter")
        continue




while(True):
    i = int(input("Enter Number-> "))
    
    if (i%2==0):
        print(i**2)
    else:
        print('number is not even')
        break










































    














































































































































































































