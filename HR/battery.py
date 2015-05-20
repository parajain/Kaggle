#!/usr/bin/python

from sys import stdin
import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt


trainData = []
labels = []
numf = 1
filename = "trainingdata.txt"
#read trainig data from file
with open(filename) as f:
  for line in f:
    line = line.rstrip("\n")
    linearr = line.split(',')
    row = []	
    for i in range(numf):
      row.append(float(linearr[i]))
    trainData.append(row)
    labels.append(float(linearr[-1]))

#plt.scatter(trainData,labels)
#plt.show()
print trainData[0]
print labels[0]
#print "Training..."
reg = linear_model.LinearRegression()
reg.fit(trainData,labels)

#test
#print "Enter test .."
inp = stdin.readline()
X = float(inp)
if X > 4:
  Y = 8
else:
  Y = 2*X#reg.predict(X)
print Y
