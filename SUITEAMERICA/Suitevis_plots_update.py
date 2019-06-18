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
from numpy import median
from collections import OrderedDict
from operator import itemgetter

suite = pd.read_csv("Final_data.csv")
suite.head()
suite.tail()

def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100

def Plot_Top(Column, heading1, heading2, col):
    Types = list(suite['SuiteSizeCode'])
    Count = sorted([[s,Types.count(s)] for s in set(Types)], key=itemgetter(1))[::-1]
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
    bar1.set_xticklabels(bar1.get_xticklabels(),rotation=90)
    bar1.set_title(heading1)
    
    ax = plt.gca()
    y_max = max(Count)
    ax.set_ylim([0, roundup(y_max)])
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')
	#plt.show()
	
    # PLOT 2
    plt.figure()
    bar2 = sns.barplot(x=Types,y=Avgs,color=col)
    bar2.set_xticklabels(bar2.get_xticklabels(),rotation=90)
    bar2.set_title(heading2)
    
    ax = plt.gca()
    y_max = max(Avgs)
    ax.set_ylim([0, roundup(y_max)])
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),fontsize=12, color='black', ha='center', va='bottom')
        
	#plt.show()


Plot_Top('DailyRent', 'Total Rents wrt Suite Code', 'Avg. Rents wrt Suite Code', 'b')


Plot_Top('Stay', 'Total Stays wrt Suit', 'Avg. Stays wrt Suite Code', 'r')

