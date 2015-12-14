import json
import pandas as pd
import numpy as np
from helper import *
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

f = open('../data/train.json').read();
trainData = json.loads(f)
X_train = [x['ingredients'] for x in trainData]
Y_train = [y['cuisine'] for y in trainData]

f = open('../data/test.json').read()
testData = json.loads(f)
X_test = [x['ingredients'] for x in testData]
id_test = [i['id'] for i in testData]


print(X_train[0]) 
print(Y_train[0])
print(trainData[0])

print('Word2vec')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
min_word_count = 1   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 20          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
print len(X_train)
#all_train = list(X_train)
#all_train.append(Y_train)

#print len(X_train)
#print len(Y_train)
#print len(all_train)

model = word2vec.Word2Vec(X_train, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context)
print 'Model word2vec completed'
print len(model['grape tomatoes'])
#print len(model['greek'])
Y_train_set = list(set(Y_train))
Y_train_class =np.zeros(len(Y_train), dtype='float32')
for i in range(0, len(Y_train)):
    Y_train_class[i] = Y_train_set.index(Y_train[i])

print Y_train_set[0]
print "Number of labels:" , len(Y_train_set)
##########################################
# Define a function to create bags of centroids
#
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5
# Initalize a k-means object and use it to extract centroids
print "Running K means"
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.index2word, idx ))

# ****** Create bags of centroids
#
# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( ( len(X_train), num_clusters),dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in X_train:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros(( len(X_test), num_clusters), dtype="float32" )
counter = 0
for review in X_test:
    test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1
'''
from sklearn import svm
clf = svm.SVC()
clf.fit(train_centroids, Y_train_class)

pred = clf.predict(test_centroids)

res = []
for i in range(0, len(X_test)):
    res.append(Y_train_set[pred[i].astype(int)])
#print res

output = pd.DataFrame(data={"id":id_test,"cuisine":res})
output.to_csv( "svm.csv", index=False, quoting=3 )


'''
import xgboost as xgb
xg_train = xgb.DMatrix( train_centroids, label=Y_train_class)
xg_test = xgb.DMatrix(test_centroids)

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

res = []
for i in range(0, len(X_test)):
    res.append(Y_train_set[pred[i].astype(int)])
#print res

output = pd.DataFrame(data={"cuisine":res,"id":id_test})
output.to_csv( "cxb.csv", index=False, quoting=3 )
'''
##########################################
print len(train_centroids)
print len(train_centroids[0])
#Y_train_class = np.transpose(Y_train_class)
print test_centroids.shape


forest = RandomForestClassifier(n_estimators = 200)

# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,Y_train_class)
pred = forest.predict(test_centroids)



res = []
for i in range(0, len(X_test)):
    res.append(Y_train_set[pred[i].astype(int)])
#print res

output = pd.DataFrame(data={"id":id_test,"cuisine":res})
output.to_csv( "boc.csv", index=False, quoting=3 )

print "Wrote BagOfCentroids.csv"

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(X_trainfeatures)  
X_trainfeatures = scaler.transform(X_trainfeatures) 
X_test_features = scaler.transform(X_test_features)

####################################XGBOOST################################
import xgboost as xgb
xg_train = xgb.DMatrix( X_trainfeatures, label=Y_train_class)
xg_test = xgb.DMatrix(X_test_features)

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
num_round = 150
bst = xgb.train(param, xg_train, num_round);
# get prediction
pred = bst.predict( xg_test );

res = []
for i in range(0, len(X_test)):
    #p = clf.predict(X_test_features[i])
    res.append(Y_train_set[pred[i].astype(int)])
#print res

output = pd.DataFrame(data={"cuisine":res,"id":id_test})
output.to_csv( "out.csv", index=False, quoting=3 )
'''
