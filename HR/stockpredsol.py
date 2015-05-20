\#!/usr/bin/py

import numpy as np
import math
import cPickle as pickle
import os

def sell(owned, prices):
    last = prices[-1]
    prices = np.array(prices[:-1])

    if last >= prices.mean():
        return owned

    return 0

def buy(m, prices):
    #std = prices.std()

    last = prices[-1]
    prices = np.array(prices[:-1])

    if last <= prices.mean():
        #if std > 2:
        return int(m / int(math.ceil(last)))

    return 0

# Taken from here:
# http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_apply(fun, a, w):
    r = np.empty(a.shape)
    r.fill(np.nan)

    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i-w+1):i+1])

    return r

#def smooth_prices(prices):
#    #print prices
#    sprices = np.array([np.hstack((a[:9], rolling_apply(np.mean, a, 10))) for a in prices])
#    return sprices
#    #return prices

def stochastic_oscillator(prices, period):
    min_p = prices[-period:].min()
    max_p = prices[-period:].max()

    if min_p == max_p:
        return 0.

    return abs(100. * (prices[-1] - min_p) / (max_p - min_p))

# Head ends here
def printTransactions(m, k, d, name, owned, prices):
    output = []

    prices = np.array(prices)
    #sprices = smooth_prices(prices)

    deviations = prices.std(1)
    #deviations = sprices.std(1)

    #sold = np.zeros(k)
    for i in reversed(np.argsort(deviations)):
        sa = stochastic_oscillator(prices[i], 3)

        if sa >= 80. and owned[i]:
            output.append("%s %s %s" % (
                name[i], 'SELL', owned[i]))

        elif sa <= 20. and m:
            num = int(m / int(math.ceil(prices[i][-1])))
            if num:
                output.append("%s %s %s" % (
                    name[i], 'BUY', num))
                m -= (num * int(math.ceil(prices[i][-1])))

        ##if name[i] == 'RIT':
        #if name[i] == 'UCSC':
        #    print '-----------'
        #    print stochastic_oscillator(prices[i], 3)
        #    print prices[i]

        #num = 0
        #if owned[i]:
        #    num = sell(owned[i], prices[i])
        #    if num:
        #        output.append("%s %s %s" % (
        #            name[i], 'SELL', num))
        #if num == 0:
        #    num = buy(m, prices[i])
        #    if num:
        #        output.append("%s %s %s" % (
        #            name[i], 'BUY', num))
       #         m -= (num * int(math.ceil(prices[i][-1])))

    return output

def parse_step():
    m, k, d = [float(i) for i in raw_input().strip().split()]

    k = int(k)
    d = int(d)
    names = []
    owned = []
    prices = []
    for data in range(k):
        temp = raw_input().strip().split()
        names.append(temp[0])
        owned.append(int(temp[1]))
        prices.append([float(i) for i in temp[2:7]])

    return m, k, d, names, owned, prices

def load_history():
    prices_history = {}
    try:
        with open('prices_history.pkl', 'r') as f:
            prices_history = pickle.load(f)
    except IOError:
        pass

    return prices_history

def save_history(k, d, names, prices):
    prices_history = {}

    for i in range(k):
        prices_history[names[i]] = prices[i]

    prices_history['_days'] = d

    with open('prices_history.pkl', 'w') as f:
        pickle.dump(prices_history, f)

def add_history(k, d, names, prices):
    prices_history = load_history()

    if prices_history and prices_history['_days'] > d:
        for i in range(k):
            hprices = prices_history.get(names[i])
            if hprices:
                hprices.append(prices[i][-1])
                prices[i] = hprices

    return prices

def remove_history():
    try:
        os.remove('prices_history.pkl')
    except:
        pass

# Tail starts here
if __name__ == '__main__':

    m, k, d, names, owned, prices = parse_step()
    #prices = add_history(k, d, names, prices)

    output = printTransactions(m, k, d, names, owned, prices)

    #save_history(k, d, names, prices)

    print len(output)
    for line in output:
        print line

