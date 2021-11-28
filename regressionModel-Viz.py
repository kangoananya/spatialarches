import os
import pandas
from scipy.sparse import data
import sklearn
import sktools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#add comment
#data viz from : 

#IO
PATH = os.path.dirname(__file__)
FILE_IN = os.path.join(PATH,'bigData.csv')
FILE_TEST = os.path.join(PATH,'theghostOfGaudi-200x200-1.25.csv')
dataset = pandas.read_csv(FILE_IN)

#actual data set used for regression
X = dataset[['Width','Depth','aspeed']]
y = dataset['Deviation']

#segregated data for visualisation
u = dataset['aspeed']
v = dataset['Width']
w = dataset['Depth']

x_pred = np.linspace(90, 170, 50)   # range of Width
y_pred = np.linspace(20, 65, 50)  # range of Depth
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

x_,y_,z_ = [],[],[]
for i,dev in enumerate(y):
    if dev < 2 and dev>-2:
        x_.append(v[i])
        y_.append(w[i])
        z_.append(u[i])

#create data for viz
U = np.array([x_,y_]).T
_v = np.array(z_)

#fake regression to match viz
regr = linear_model.LinearRegression()
regr.fit(U,_v)

predicted = regr.predict(model_viz)

print(regr.coef_)
print(regr.intercept_)
fig = plt.figure(figsize=(15,15))
ax2 = fig.add_subplot(111,projection = '3d')
ax2.set_xlabel('Width', fontsize=12)
ax2.set_ylabel('Depth', fontsize=12)
ax2.set_zlabel('Extruder Speed', fontsize=12)

ax2.plot(x_, y_, z_, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
ax2.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=10, edgecolor='xkcd:deep pink')

#for ii in np.arange(0, 360, 1):
#    ax2.view_init(elev=32, azim=ii)
#    fig.savefig('gif_image%d.png' % ii)

#actual regression and deviation calculation
q_reg = make_pipeline((PolynomialFeatures(1)),linear_model.LinearRegression())
test_speeds = [n/100 for n in range(100,225,5)]
q_reg.fit(X,y)


#run prediction on test data
test_data = pandas.read_csv(FILE_TEST)
w_ = test_data['Width']
d_ = test_data['Depth']
a_ = test_data['aspeed']

prediction = q_reg.predict(test_data[['Width','Depth','aspeed']])

for i,p in enumerate(prediction):
    if p<1.5 and p>-1.5:
        print("for Width %s and Depth %s, best aspeed would be %s" %(w_[i],d_[i],a_[i]))
#save least value as list
