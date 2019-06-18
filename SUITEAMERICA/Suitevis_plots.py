
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import collections
from operator import itemgetter
from collections import OrderedDict
from operator import itemgetter
from matplotlib.pyplot import show
from pandas import DataFrame
import seaborn as sns
from seaborn import FacetGrid
get_ipython().magic('matplotlib inline')
from numpy import median


# In[2]:


os.chdir("/Users/chanduboss/Desktop/SUITEAMERICA")
os.getcwd()


# In[3]:


suite = pd.read_csv("Final_data.csv")
suite.head()
suite.tail()


# In[4]:


suite.dtypes


# In[5]:


suite.DailyRent = suite.DailyRent.astype(int)
suite.Stay = suite.Stay.astype(int)
suite.MoveIn_Week = suite.MoveIn_Week.astype(int)


# In[6]:


suite.info()


# In[7]:


bar = sns.barplot(x="SuiteSizeCode",y="DailyRent",data= suite,color="b")
bar.set_xticklabels(bar.get_xticklabels(),rotation=90)
bar.set_title("Average prices of the Suites")



# In[8]:


miy = sns.factorplot("MoveinYear", data=suite, aspect=1.5, kind="count", color="red")
miy.set_xticklabels(rotation=30)
    
def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

ax = plt.gca()
y_max = suite['MoveinYear'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[9]:


moy = sns.factorplot("MoveOutYear", data=suite, aspect=1.5, kind="count", color="red")
moy.set_xticklabels(rotation=40)

def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

ax = plt.gca()
y_max = suite['MoveOutYear'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[10]:


mim = sns.factorplot("Movein_Month", data=suite, aspect=2, kind="count", color="red",
                     order=["January","February","March","April","May","June","July","August",
                           "September","October","November","December"])
mim.set_xticklabels(rotation=90)

def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

ax = plt.gca()
y_max = suite['Movein_Month'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[11]:




def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

mom = sns.factorplot("Moveout_Month", data=suite, aspect=2, kind="count", color="red",
                    order=["January","February","March","April","May","June","July","August",
                           "September","October","November","December"])
# mom.set_xticklabels(rotation=90)

ax = plt.gca()
y_max = suite['Moveout_Month'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[12]:


miw = sns.factorplot("MoveIn_Week", data=suite, aspect=4, kind="count", color="red")
miw.set_xticklabels(rotation=90)

def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

ax = plt.gca()
y_max = suite['MoveIn_Week'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[13]:


mow = sns.factorplot("MoveOut_Week", data=suite, aspect=4, kind="count", color="red")
mow.set_xticklabels(rotation=90)


def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

ax = plt.gca()
y_max = suite['MoveOut_Week'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[14]:


mid = sns.factorplot("Movein_Day", data=suite, aspect=2, kind="count", color="red",
                     order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
mid.set_xticklabels(rotation=90)

def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

ax = plt.gca()
y_max = suite['Movein_Day'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[15]:


mod = sns.factorplot("Moveout_Day", data=suite, aspect=3, kind="count", color="red",
                     order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
mod.set_xticklabels(rotation=90)

def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

ax = plt.gca()
y_max = suite['Moveout_Day'].value_counts().max()
ax.set_ylim([0, roundup(y_max)])

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[16]:


zone = sns.barplot(x="ZoneCode",y="DailyRent",data= suite,color="b")
zone.set_xticklabels(zone.get_xticklabels(),rotation=90)
zone.set_title("Daily Rents per zone")


# In[17]:


sns.distplot(suite["DailyRent"],hist=True)
sns.distplot(suite['Stay']);


# In[18]:


suite.SuiteSizeCode.dtype


# In[19]:


suite.SuiteSizeCode = np.arange(len(suite.SuiteSizeCode))
plt.bar(suite.SuiteSizeCode, height=1, color=(0.2, 0.4, 0.6, 0.6))


# In[20]:


from collections import OrderedDict
from operator import itemgetter


# In[26]:


def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

def Plot_Top(Column, heading1, heading2, col):
    Types = list(suite['SuiteSizeCode'])
    Count = sorted([[s,Types.count(s)] for s in set(Sizes)], key=itemgetter(1))[::-1]
    Top = Count[:6]
    Types = [t[0] for t in Top]
    Totals = [t[1] for t in Top]
    # print(Types)

    Map = OrderedDict()
    for i in range(suite.shape[0]):
        code = suite.iloc[i]['SuiteSizeCode']
        count = suite.iloc[i][Column]
        if code in Types:
            if code in Map.keys():
                Map[code]+=count
            else:
                Map[code]=count
    print(Map)

    Types = list(Map.keys())
    Count = list(Map.values())

    # print(Count)
    # print(Totals)
    Avgs = []
    for i in range(len(Count)):
        avg = Count[i]/Totals[i]
        Avgs.append(avg)
    # print(Avgs)

    # PLOT 1
    plt.figure()
    bar1 = sns.barplot(x=Types,y=Count,color=col)
    bar1.set_xticklabels(bar.get_xticklabels(),rotation=90)
    bar1.set_title(heading1)
    
    ax = plt.gca()
    y_max = max(Count)
    ax.set_ylim([0, roundup(y_max)])
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')

    # PLOT 2
    plt.figure()
    bar2 = sns.barplot(x=Types,y=Avgs,color=col)
    bar2.set_xticklabels(bar.get_xticklabels(),rotation=90)
    bar2.set_title(heading2)
    
    ax = plt.gca()
    y_max = max(Avgs)
    ax.set_ylim([0, roundup(y_max)])
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')


# In[27]:


print(Avgs)


# In[25]:


print(Map)


# In[28]:


Plot_Top('DailyRent', 'Total Rents wrt Suite Code', 'Avg. Rents wrt Suite Code', 'b')


# In[29]:


Plot_Top('Stay', 'Total Stays wrt Suit', 'Avg. Stays wrt Suite Code', 'r')

