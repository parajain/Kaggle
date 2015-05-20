#!/usr/bin/python

from sys import stdin

inp = stdin.readline().strip()
inpsplit = inp.split()
m = float(inpsplit[0])
k = int(inpsplit[1])
d = int(inpsplit[2])


names = []
owned = []
prices = []
predicted = []

for data in range(k):
    temp = stdin.readline().strip().split()
    names.append(temp[0])
    owned.append(int(temp[1]))
    prices.append([float(i) for i in temp[2:7]])
    predicted.append(sum(prices[data])/5)


diff = []
for i in range(k):
    p = prices[i]
    #print p[-1]
    #print avg[i]
    diff.append((predicted[i] - p[-1],int(i)))
   
#print diff
diff.sort(reverse = True)
#print diff
#if d == 0 , sell everything??

transaction = []
numtr = 0

for i in range(k):
    d = diff[i]
    if d[0] == 0:
        continue
    if d[0] > 0:
        p = prices[d[1]]
        u = int(m/p[-1])
        if u > 0:
            numtr = numtr + 1
            transaction.append(names[d[1]] + " BUY " + str(u))
            m = m - u*p[-1]
            owned[d[1]] = owned[d[1]] + u
    elif owned[d[1]] > 0:
        numtr = numte + 1
        transaction.append(names[d[1]] + " SELL " + owned[d[1]])
        owned[d[1]] = 0


print numtr
if numtr > 0:
    for t in transaction:
        print t



