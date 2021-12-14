import os
import pandas
from scipy.sparse import data
import sklearn
import sktools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#add comment
#data viz from : 

#IO
PATH = os.path.dirname(__file__)
FILE_IN = os.path.join(PATH,'bigData.csv')
dataset = pandas.read_csv(FILE_IN)

#actual data set used for regression
width = dataset['Width']
height = dataset['Depth']

x = []
for i in range(len(width)):
    x.append(float(width[i])/float(height[i]))

y = dataset['Deviation']

z = dataset['aspeed']
u = dataset['Width']
v = list(zip(z,u))

v_ = set(v)

v_ = [(x,y) for x,y in v_ if pandas.isnull(x) == False]
print (v_)



def draw(x_,y_,aspeed,cv):
    fig = plt.figure(figsize=(15,10))
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel('Width/Height Ratio', fontsize=20)
    ax2.set_ylabel('Deviation', fontsize=20)
    ax2.set_title('Aspeed = '+str(aspeed)+'   CurveWidth = '+str(cv) ,fontsize=30)
    ax2.scatter(x_, y_, color="black", s = 30)

    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_position(('data',0))
    ax2.spines['top'].set_color('grey')
    ax2.spines['top'].set_linewidth(3)
    
    ax2.grid(linewidth=0.2, alpha=1)
    ax2.plot(x_, y_, color='xkcd:deep pink', linewidth=1.5)

    fig.savefig('image%scurvewidth%saspeed.jpg'%(cv,aspeed))



for v__ in v_:
    x_,y_ = [],[]
    if v__ == (2.0,100):
        for i,vs in enumerate(v):
            if vs == v__:
                x_.append(x[i])
                y_.append(y[i])
        s,cv = v__
        if len(x_) > 2:
            draw(x_,y_,s,cv)
    else:
        for i,vs in enumerate(v):
            if vs == v__:
                x_.append(x[i])
                y_.append(y[i])
        s,cv = v__
        if len(x_) > 3:
            a = list(zip(x_,y_))
            a = sorted(a)
            x_ = [x for x,y in a]
            y_ = [y for x,y in a]
            draw(x_,y_,s,cv)


'''

print(regr.coef_)
print(regr.intercept_)
fig = plt.figure(figsize=(15,15))
ax2 = fig.add_subplot(111)
ax2.set_xlabel('Width/Height Ratio', fontsize=12)
ax2.set_ylabel('Deviation', fontsize=12)


ax2.plot(x_, y_, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
#ax2.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=10, edgecolor='xkcd:deep pink')

#for ii in np.arange(0, 360, 1):
#    ax2.view_init(elev=32, azim=ii)
#    fig.savefig('gif_image%d.png' % ii)

'''