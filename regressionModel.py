import pandas
import sklearn
import sktools
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


dataset = pandas.read_csv("C:/Users/ananya/Documents/masdfab-notes/work/T1-MiniProject/bigData.csv")

X = dataset[['Width','Depth','aspeed']]
y = dataset['Deviation']

u = dataset['aspeed']
v = dataset['Width']
w = dataset['Depth']


regr = linear_model.LinearRegression()
regr.fit(X,y)

#have test data with width and height and try list of speeds
q_reg = make_pipeline((PolynomialFeatures(2)),linear_model.LinearRegression())
test_speeds = [n/100 for n in range(100,225,5)]
q_reg.fit(X,y)
#for loop with prediction for test data as data which is needed
#have prediction generator and save least value

test_data = pandas.read_csv("C:/Users/ananya/Documents/masdfab-notes/work/T1-MiniProject/theghostOfGaudi.csv")
w_ = test_data['Width']
d_ = test_data['Depth']
prediction = q_reg.predict([[135,49,1.8]])
#print(prediction)

for w,d in zip(w_,d_):
    for s in test_speeds:
        prediction = q_reg.predict([[w,d,s]])
        if(prediction>-1 and prediction<1):
            print('%s width and %s depth at %s aspeed - %s deviation' %(w,d,s,prediction))


#save least value as list
