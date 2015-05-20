#!/usr/bin/python
from numpy import *
from sys import stdin
#F is the number of observed features. 
#N is the number of rows for which features as well as price per square-foot have been noted.

line = stdin.readline()
linesplit = line.split()
numf = int(linesplit[0])
n = int(linesplit[1])
print "features = " +str(numf) + " n = " + str(n)

trainData = []
labels = []


#read training
for i in range(n):
  train  = stdin.readline()
  s = train.split()
  #print s
  r = []
  r.append(1)
  for f in range(numf):
    r.append(float(s[f]))
  trainData.append(r)
  labels.append(float(s[-1]))

#train the data
xMat = mat(trainData)
xTx = xMat.T * xMat
yMat = mat(labels).T
ws = xTx.I * (xMat.T*yMat)
    
#predict
numTest = int(stdin.readline())
#print(numTest)
for w in range(numTest):
    train = stdin.readline()
    trainD = train.split()
    raw = []
    raw.append(1)
    for d in range(numf):
        raw.append(float(trainD[d]))
        
    out =  mat(raw) * (ws)
    print(out.item(0))
