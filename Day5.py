# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:26:21 2022

@author: vikas
"""




#Heat Map

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


uniform_data = np.random.rand(10, 10)
uniform_data.shape

df = pd.DataFrame(uniform_data)
df


fig, ax = plt.subplots()
hm = ax.pcolor(df, cmap = plt.cm.Reds)
plt.colorbar(hm)


import matplotlib.pyplot as plt
plt.figure(dpi=600)
import seaborn as sns
ax= sns.heatmap(df, annot=True,fmt=".1f",
                xticklabels=['A','B','C','D'],
                yticklabels=['A','B','C','D'],
                cbar=False,)

x1 = np.random.randint(1,10, size=10)
x2 = np.random.randint(100,200, size=10)


df = pd.DataFrame({'X':x1, 'Y':x2})



import matplotlib.pyplot as plt
plt.figure(dpi=600)
import seaborn as sns
ax= sns.heatmap(df.corr(), annot=True,fmt=".1f",
                xticklabels=['A','B'],
                yticklabels=['C','D'],
                cbar=False,)



import matplotlib.pyplot as plt
import numpy as np

from pydataset import data
mt = data('mtcars')

plt.hist(mt.mpg)


import seaborn as sns

sns.distplot(mt.mpg)





import matplotlib.pyplot as plt
labels = ['Male',  'Female']
percentages = [60, 40]


explode=(0.15,0)
#

color_palette_list = ['#f600cc', '#ADD8E6', '#63D1F4', '#0EBFE9', '#C1F0F6', '#0099CC']

fig, ax = plt.subplots()

ax.pie(percentages, explode=explode, labels=labels, 
       colors= color_palette_list, autopct='%.1f%%',  
       shadow=True, startangle=110,  pctdistance=1.2, 
       labeldistance=1.4)
ax.axis('equal')
ax.set_title("Distribution of Gender in Class", y=1)
ax.legend(frameon=False, bbox_to_anchor=(0.2,0.8))
plt.show()




#Case Study


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ww = pd.read_csv('winequality-white.csv', sep=';')
ww

rw = pd.read_csv('winequality-red.csv')
rw

rw.head(1)
rw.columns

ww['type'] = 'White'
rw['type'] = 'Red'

wine = pd.concat([rw,ww])
wine['quality'] = wine['quality'].astype('category')
wine.dtypes

'''
pd.Categorical(wine['quality'], categories=('low','high'),ordered = True)
wine['level']
'''


fn = lambda value:'low' if value <= 5 else 'medium' if value <= 7 else 'high'
wine['level'] = wine['quality'].apply(fn)
wine['level']


wine.describe()
np.sum(wine.isnull())

wines = wine.sample(frac=1)
wine

wines.columns


wines['fixed acidity'].hist(color='steelblue', 
                            edgecolor='black', linewidth=1.0,
                            xlabelsize=8, ylabelsize=8, grid=False)    

plt.tight_layout(rect=(0, 0, 1.2, 1.2))   



fig = plt.figure(figsize = (6,4), dpi=300)
fig

title = fig.suptitle("Sulphates Content in Wine", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Sulphates")
ax.set_ylabel("Frequency") 

ax.text(1.2, 10, r'$\lambda$='+str(round(wines['sulphates'].mean(),2)), 
         fontsize=12)

freq, bins, patches = ax.hist(wines['sulphates'], color='steelblue', bins=30,
                                    edgecolor='black', linewidth=1)
plt.show()





import seaborn as sns

# Density Plot
fig = plt.figure(figsize = (6, 4), dpi=300)
title = fig.suptitle("Sulphates Content in Wine", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,1, 1)
ax1.set_xlabel("Sulphates")
ax1.set_ylabel("Frequency") 
sns.distplot(wines['sulphates'], kde=True)









# Bar Plot
fig = plt.figure(figsize = (6, 4), dpi=300)
title = fig.suptitle("Wine Quality Frequency", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Quality")
ax.set_ylabel("Frequency") 

w_q = wines['quality'].value_counts()

w_q

w_q = (list(w_q.index), list(w_q.values))

w_q


ax.tick_params(axis='both', which='major', labelsize=8.5)
bar = ax.bar(w_q[0], w_q[1], color='steelblue', 
        edgecolor='black', linewidth=1)




# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(10, 6), dpi=300)

corr = wines.corr()
corr
round(corr,2)

hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)




from pydataset import data
mtcars = data('mtcars')

mtcars.dtypes

mtcars.corr()

mtcars.columns

import seaborn as sns

mtcars = mtcars[['mpg', 'cyl', 'disp','hp', 'drat']]

pp = sns.pairplot(mtcars, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)




wines.columns
# facets with histograms

fig = plt.figure(figsize = (10,4), dpi=300)
title = fig.suptitle("Sulphates and Chlorides Content in Wine", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,2, 2)
ax1.set_title("Wine")
ax1.set_xlabel("Sulphates")
ax1.set_ylabel("Frequency") 
ax1.set_ylim([0, 1000])
ax1.text(1.2, 800, r'$\mu$='+str(round(wines['sulphates'].mean(),2)), 
         fontsize=12)
r_freq, r_bins, r_patches = ax1.hist(wines['sulphates'], color='red', bins=15,
                                     edgecolor='black', linewidth=1)

ax2 = fig.add_subplot(1,2, 1)
ax2.set_title("Wine")
ax2.set_xlabel("Chlorides")
ax2.set_ylabel("Frequency")
ax2.set_ylim([0, 1200])
ax2.text(0.8, 800, r'$\mu$='+str(round(wines['chlorides'].mean(),2)), 
         fontsize=12)
w_freq, w_bins, w_patches = ax2.hist(wines['chlorides'], color='white', bins=15,
                                     edgecolor='black', linewidth=1)








































