import xgboost as xgb
import numpy as np

X_train = np.loadtxt('features_train.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv')
X_test = np.loadtxt('features_test.csv', delimiter=',')
xg_train = xgb.DMatrix( X, label=Y)
xg_test = xgb.DMatrix(X_test)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.2
param['max_depth'] = 25
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 20

#watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 100
bst = xgb.train(param, xg_train, num_round);
# get prediction
pred = bst.predict( xg_test );

outfile = open('output.csv', 'a')

res = []
for i in range(0, len(X_test)):
    outfile.write("{},{}".format(i, pred[i]))
    res.append(Y_train_set[pred[i].astype(int)])
#print res
