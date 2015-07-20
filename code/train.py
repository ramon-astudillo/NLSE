#!/usr/bin/python
'''
Sub-space training 
'''
import sys
import os
sys.path.append('code')
import shutil
import re
import numpy as np
import theano
import theano.tensor as T
import cPickle
#
import nlse
import Print
#
from Fmeasure import FmesSemEval

# DEBUGGING
from ipdb import set_trace

####################################
#           CONFIG 
####################################

# TRAIN 
n_iter  = 8
lrate   = np.array(0.01).astype(theano.config.floatX)

# MODEL 
# Pre-trained embeddings
emb_path = 'DATA/features/wang/struc_skip_600.pkl'
# Sub space size 
sub_size = 10
# Model name
model_path = 'DATA/models/skip600_s10_iter8_lrate001.pkl' 

# Create or load model
nn = nlse.NLSE(emb_path, sub_size=sub_size, hidden=hidden)

####################################
#           SGD TRAIN 
####################################

# SEMEVAL TRAIN TWEETS
print "Loading %s" % train_data 
print "Training %s" % model_path
with open(train_data, 'rb') as fid:
    [word2idx, idx2wrd, train_x, train_y, dev_x, dev_y] = cPickle.load(fid) 

# RESHAPE TRAIN DATA AS A SINGLE NUMPY ARRAY
# Start and end indices
lens = np.array([len(tr[0]) for tr in train_x]).astype('int32')
st   = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype('int32')
ed   = (st + lens).astype('int32')
x    = np.zeros((ed[-1], 1))
for i, ins_x in enumerate(train_x):
    x[st[i]:ed[i]] = ins_x[0][:, None].astype('int32') 

# FUNCTION FOR BATCH UPDATE
# Train data and instance start and ends
x  = theano.shared(x.astype('int32'), borrow=True) 
y  = theano.shared(np.array(train_y).astype('int32'), borrow=True)
st = theano.shared(st, borrow=True)
ed = theano.shared(ed, borrow=True)
# SGD Update rule
updates = [(pr, pr-lrate*gr) for pr, gr in zip(nn.params, nn.nablas)]
# Mini-batch
i  = T.lscalar()
givens={ nn.z0 : x[st[i]:ed[i], 0],
         nn.y  : y[i] }
train_batch = theano.function(inputs=[i], outputs=nn.F, updates=updates, 
                              givens=givens)

# EPOCH LOOP
last_cr  = None
best_cr  = [0, 0]
for i in np.arange(n_iter):
    p_train = 0 
    # BATCH LOOP
    for j in np.arange(len(train_x)).astype('int32'): 
        p_train += train_batch(j) 
        # INFO
        if not (j % 100):
            sys.stdout.write("\rTraining %d/%d" % (j+1, len(train_x)))
            sys.stdout.flush()   

    # Evaluation
    cr      = 0.
    mapp    = np.array([ 1, 2, 0])
    ConfMat = np.zeros((3, 3))
    for j, x, y in zip(np.arange(len(dev_x)), dev_x, dev_y):
        # Prediction
        p_y   = nn.forward(x[0])
        hat_y = np.argmax(p_y)
        # Confusion matrix
        ConfMat[mapp[y[0]], mapp[hat_y]] += 1
        # Accuracy
        cr    = (cr*j + (hat_y == y[0]).astype(float))/(j+1)
        # INFO
        sys.stdout.write("\rDevel %d/%d            " % (j+1, len(dev_x)))
        sys.stdout.flush()   
    # Compute SemEval scores
    Fm = FmesSemEval(ConfMat)

    # INFO
    if last_cr:
        # Keep bet model
        if best_cr[0] < cr:
            best_cr = [cr, i+1]
        delta_cr = cr - last_cr
        if delta_cr >= 0:
            print ("\rEpoch %2d/%2d: Acc %2.5f%% \033[32m+%2.5f\033[0m (Fm %2.5f%%)" % 
                   (i+1, n_iter, cr*100, delta_cr*100, Fm*100))
        else: 
            print ("\rEpoch %2d/%2d: Acc %2.5f%% \033[31m%2.5f\033[0m (Fm %2.5f%%)" % 
                   (i+1, n_iter, cr*100, delta_cr*100, Fm*100))
    else:
        print "\rEpoch %2d/%2d: %2.5f (Fm %2.5f%%)" % (i+1, n_iter, cr*100,
                                                       Fm*100)
        best_cr = [cr, i+1]
    last_cr = cr

    # SAVE MODEL
    tmp_model_path = model_path.replace('.pkl','.%d.pkl' % (i+1))
    nn.save(tmp_model_path)

# Store best model with the original model name
tmp_model_path = model_path.replace('.pkl','.%d.pkl' % best_cr[1])
print "Best model %s -> %s\nDev %2.5f %%" % (tmp_model_path, 
                                             model_path, best_cr[0]*100)
shutil.copy(tmp_model_path, model_path)
