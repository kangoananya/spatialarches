import os
import pandas
import sklearn
import sktools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#add comment

PATH = os.path.dirname(__file__)
FILE_IN = os.path.join(PATH,'biggerData.csv')
FILE_TEST = os.path.join(PATH,'theghostOfGaudi.csv')
dataset = pandas.read_csv(FILE_IN)

x_ = dataset['Width']
xx_ = dataset['Curve Length']
X = xx_/x_
X = np.array(X)
X = X.reshape(-1,1)
#X = dataset[['Width','Curve Length']]
y = dataset['Depth']
y = np.array(y)


regr = linear_model.LinearRegression()
regr.fit(X,y)

print('%s (width) + %s (curve length)  + (%s)' %(regr.coef_,'no',regr.intercept_))
