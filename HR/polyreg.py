#!/usr/bin/python
from numpy import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sys import stdin
#F is the number of observed features. 
#N is the number of rows for which features as well as price per square-foot have been noted.

line = stdin.readline()
linesplit = line.split()
numf = int(linesplit[0])
n = int(linesplit[1])
#print "features = " +str(numf) + " n = " + str(n)

trainData = []
labels = []
testData = []
#testlabels = []
#read training
for i in range(n):
  train  = stdin.readline()
  s = train.split()
  #print s
  r = []
  #r.append(1)
  for f in range(numf):
    r.append(float(s[f]))
  trainData.append(r)
  labels.append(float(s[-1]))

#read test
numTest = int(stdin.readline())
#print "num of test = " + str(numTest)
for w in range(numTest):
  t = stdin.readline()
  ts = t.split()
  raw = []
  for d in range(numf):
    raw.append(float(ts[d]))
  testData.append(raw)

#train the data polynomial
poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(trainData)
Xtest= poly.fit_transform(testData)

reg = linear_model.LinearRegression()
reg.fit(X,labels)
predict = reg.predict(Xtest)

for o in range(numTest):
  print predict[o]


