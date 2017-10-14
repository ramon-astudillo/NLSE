'''
SemEval models
'''

import numpy as np
import dropout
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

# DEBUGGING
from ipdb import set_trace

def init_W(size, rng, init=None, shared=True):
    '''
    Random initialization
    '''
    if len(size) == 2:
        n_out, n_in = size
    else:
        n_out, n_in = size[0], size[3]
    # Uniform init scaling
    if init == 'glorot-tanh':    
        w0 = np.sqrt(6./(n_in + n_out))   
    elif init == 'glorot-sigmoid':    
        w0 = 4*np.sqrt(6./(n_in + n_out))   
    else:
        w0 = init
    W = np.asarray(rng.uniform(low=-w0, high=w0, size=size))

    if shared:
        return theano.shared(W.astype(theano.config.floatX), borrow=True)
    else:
        return W.astype(theano.config.floatX)

def forward(z0, params, init, dropout_prob, train=False):

    # Get nicer names for parameters
    E, S, C = params

    # Use an intermediate sigmoid
    z1 = E[:, z0]                 # embedding
    # dropout
    if dropout_prob:
        z1b = dropout.dropout(z1, dropout_prob, seed=init, training=train)
    else:
        z1b = z1
    z1s = T.dot(S, z1b)   # subspace
    z2 = T.nnet.sigmoid(z1s)  
    # Hidden layer
    z3 = T.dot(C, z2)
    z4 = T.sum(z3, 1)                  # Bag of words
    hat_y = T.nnet.softmax(z4.T).T

    hat_y = T.clip(hat_y, 1e-12, 1.0 - 1e-12)

    # Naming
    z1.name = 'embedding'
    z1s.name = 'subspace'
    z2.name = 'non-linear'
    z3.name = 'sentiment-vector'
    z4.name = 'bow'
    hat_y.name = 'hat_y'

    return hat_y, z4, z3, z2, z1s, z1

class NN():
    '''
    Embedding subspace
    '''
    def __init__(self, emb_path, sub_size, model_file=None, weight_CM=None,
                 init=None, dropout_prob=0, init_sub=0.1, init_clas=0.7):

        # Random Seed
        if init is None:
            rng = np.random.RandomState(1234)        
        else:
            rng = np.random.RandomState(init)        

        if model_file:
            # Check conflicting parameters given 
            if emb_path is not None or sub_size is not None:
                raise EnvironmentError, ("When loading a model emb_path and "
                                         "sub_size must be set to None")
            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [E, S, C, emb_path] = cPickle.load(fid)
            E = theano.shared(E, borrow=True)
            S = theano.shared(S, borrow=True)
            C = theano.shared(C, borrow=True)
            self.emb_path = emb_path
        else:
            # Embeddings e.g. word2vec.   
            with open(emb_path, 'rb') as fid:
                E = cPickle.load(fid).astype(theano.config.floatX)
            emb_size, voc_size = E.shape
            # This is fixed!
            E = theano.shared(E, borrow=True)
            # Embedding subspace projection
            S = init_W((sub_size, emb_size), rng, init=init_sub) # 0.0991
            # Hidden layer
            C = init_W((3, sub_size), rng, init=init_clas) # 0.679
            # Store the embedding path used
            self.emb_path = emb_path

        # Fixed parameters
        self.params = [E, S, C]
        # Compile
        self.compile(weight_CM, init, dropout_prob)

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def compile(self, weight_CM, init, dropout_prob):
        '''
        Forward pass and Gradients
        '''

        # FORWARD PASS
        # tweet in one hot
        self.z0 = T.ivector('tweet')                    
        self.hat_y, z4, z3, z2, z1s, z1 = forward(self.z0, self.params, init, 
                                                  0, train=False)
        # Compile 
        self.fwd = theano.function([self.z0], self.hat_y)
        
        # TRAINING COST 
        # Train cost minus log probability
        hat_y_tr, _, _, _, _, z1_tr = forward(self.z0, self.params, init, 
                                              dropout_prob, train=True)
        self.y = T.ivector('sentiment-label')                             
        self.z1 = z1_tr
        if weight_CM:
            WCM = (weight_CM[self.y, :].T)*T.log(hat_y_tr)
            self.F = -T.mean(WCM.sum(0))        
        else:
            self.F = -T.mean(T.log(hat_y_tr)[self.y])        
        self.cost = theano.function([self.z0, self.y], self.F)

        # Naming
        self.z0.name = 'z0'
        self.z1.name = 'z1'
        self.y.name = 'y'
        self.F.name = 'F'

    def save(self, model_file):
        with open(model_file, 'wb') as fid: 
            param_list = [W.get_value() for W in self.params] + [self.emb_path]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)
