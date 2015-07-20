'''
SemEval models
'''

import numpy as np
import theano
import theano.tensor as T
import cPickle

def init_W(size, rng):
    '''
    Random initialization
    '''
    if len(size) == 2:
        n_out, n_in = size
    else:
        n_out, n_in = size[0], size[3]
    w0 = np.sqrt(6./(n_in + n_out))   
    W = 4*np.asarray(rng.uniform(low=-w0, high=w0, size=size))
    return theano.shared(W.astype(theano.config.floatX), borrow=True)

class NLSE():
    '''
    Embedding subspace
    '''
    def __init__(self, emb_path, sub_size=10, model_file=None):

        # Random Seed
        rng = np.random.RandomState(1234)        
        if model_file:
            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [W1, W2, W3] = cPickle.load(fid)
            W1 = theano.shared(W1, borrow=True)
            W2 = theano.shared(W2, borrow=True)
            W3 = theano.shared(W3, borrow=True)
        else:
            # Embeddings e.g. Wang's, word2vec.   
            with open(emb_path, 'rb') as fid:
                W1 = cPickle.load(fid).astype(theano.config.floatX)
            emb_size, voc_size = W1.shape
            # This is fixed!
            W1 = theano.shared(W1, borrow=True)
            # Embedding subspace projection
            W2 = init_W((sub_size, emb_size), rng) 
            # Hidden layer
            W3 = init_W((3, sub_size), rng) 

        # Fixed parameters
        self.W1     = W1
        # Parameters to be updated 
        self.params = [W2, W3]
        # Compile
        self.compile()

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def gradients(self, x, y):
        return [gr(x.astype('int32'), y.astype('int32')) for gr in self.grads]

    def compile(self):
        '''
        Forward pass and Gradients
        '''
        # Get nicer names for parameters
        W1, W2, W3 = [self.W1] + self.params

        # FORWARD PASS
        # Embedding layer subspace
        self.z0    = T.ivector()                    # tweet in one hot

        # Use an intermediate sigmoid
        z1         = W1[:, self.z0]                 # embedding
        z2         = T.nnet.sigmoid(T.dot(W2, z1))  # subspace
        # Hidden layer
        z3         = T.dot(W3, z2)
        z4         = T.sum(z3, 1)                   # Bag of words
        self.hat_y = T.nnet.softmax(z4.T).T
        self.fwd   = theano.function([self.z0], self.hat_y)
        
        # TRAINING COST AND GRADIENTS
        # Train cost minus log probability
        self.y = T.ivector()                          # reference out
        self.F = -T.mean(T.log(self.hat_y)[self.y])   # For softmax out 
        # Update only last three parameters
        self.nablas = [] # Symbolic gradients
        self.grads  = [] # gradients
        for W in self.params:
            self.nablas.append(T.grad(self.F, W))
            self.grads.append(theano.function([self.z0, self.y], T.grad(self.F, W)))
        self.cost = theano.function([self.z0, self.y], self.F)

    def save(self, model_file):
        with open(model_file, 'wb') as fid: 
            param_list = [self.W1.get_value()] + [W.get_value() 
                          for W in self.params]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)
