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
FILE_TEST = os.path.join(PATH,'theghostOfGaudi_simple.csv')
dataset = pandas.read_csv(FILE_IN)

X = dataset[['Width','Depth','aspeed']]
y = dataset['Deviation']
print(type(y))
u = dataset['aspeed']
v = dataset['Width']
w = dataset['Depth']


regr = linear_model.LinearRegression()
regr.fit(X,y)

#have test data with width and height and try list of speeds
q_reg = make_pipeline((PolynomialFeatures(1)),linear_model.LinearRegression())
test_speeds = [n/100 for n in range(100,225,5)]
q_reg.fit(X,y)
#for loop with prediction for test data as data which is needed
#have prediction generator and save least value

test_data = pandas.read_csv(FILE_TEST)
w_ = test_data['Width']
d_ = test_data['Depth']
prediction = q_reg.predict([[62,25,2.1]])
print(prediction)

#
##for w,d in zip(w_,d_):
#    for s in test_speeds:
#        prediction = q_reg.predict([[w,d,s]])
#        if(prediction>-1 and prediction<1):
#           print('%s width and %s depth at %s aspeed - %s deviation' %(w,d,s,prediction))


#save least value as list
