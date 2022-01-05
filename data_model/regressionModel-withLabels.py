import os
import pandas
import sklearn
import sktools
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#add comment

PATH = os.path.dirname(__file__)
FILE_IN = os.path.join(PATH,'bigData.csv')
FILE_TEST = os.path.join(PATH,'test_data.csv')
dataset = pandas.read_csv(FILE_IN)

X = dataset[['Width','Depth','aspeed']]
y = dataset['Deviation']

u = dataset['aspeed']
v = dataset['Width']
w = dataset['Depth']


regr = linear_model.LinearRegression()
regr.fit(X,y)

#have test data with width and height and try list of speeds
q_reg = make_pipeline((PolynomialFeatures(1)),linear_model.LinearRegression())
test_speeds = [n/100 for n in range(100,230,5)]
q_reg.fit(X,y)
#for loop with prediction for test data as data which is needed
#have prediction generator and save least value

test_data = pandas.read_csv(FILE_TEST)
w_ = test_data['Width']
d_ = test_data['Depth']
#a_ = test_data['aspeed']

#prediction = q_reg.predict(test_data[['Width','Depth','aspeed'
lis = []

for w,d in zip(w_,d_):
    speed = []
    for s in test_speeds:
        prediction = q_reg.predict([[w,d,s]])
        if(prediction>-1 and prediction<2.5):
           print('%s width and %s depth at %s aspeed - %s deviation' %(w,d,s,prediction))
           speed.append(s)
    lis.append(speed[0])

#for i,p in enumerate(prediction):
#    speed = []
#    if p<2.5 and p>-1.5:
#        print("for Width %s and Depth %s, best aspeed would be %s" %(w_[i],d_[i],a_[i]))
 #       continue
#save least value as list
print(lis)