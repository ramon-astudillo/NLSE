'''
SemEval models
'''

import numpy as np
import theano
import theano.tensor as T
import cPickle

from ipdb import set_trace

def init_W(size, rng):
    '''
    Random initialization
    '''
    if len(size) == 2:
        n_out, n_in = size
    else:
        n_out, n_in = size[0], size[3]
    w0 = np.sqrt(6./(n_in + n_out))   
    W = np.asarray(rng.uniform(low=-w0, high=w0, size=size))
    return theano.shared(W.astype(theano.config.floatX), borrow=True)

class NLSE():
    '''
    Embedding subspace
    '''
    def __init__(self, emb_path, sub_size, model_file=None):

        # Random Seed
        rng = np.random.RandomState(1234)        
        if model_file:
            # Check conflicting parameters given 
            if emb_path is not None or sub_size is not None:
                raise EnvironmentError, ("When loading a model emb_path and "
                                         "sub_size must be set to None")
            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [E, S, C, emb_path] = cPickle.load(fid)
            E             = theano.shared(E, borrow=True)
            S             = theano.shared(S, borrow=True)
            C             = theano.shared(C, borrow=True)
            self.emb_path = emb_path
        else:
            # Embeddings e.g. word2vec.   
            with open(emb_path, 'rb') as fid:
                E = cPickle.load(fid).astype(theano.config.floatX)
            emb_size, voc_size = E.shape
            # This is fixed!
            E = theano.shared(E, borrow=True)
            # Embedding subspace projection
            S = init_W((sub_size, emb_size), rng) 
            # Hidden layer
            C = init_W((3, sub_size), rng) 
            # Store the embedding path used
            self.emb_path = emb_path

        # Fixed parameters
        self.E     = E
        # Parameters to be updated 
        self.params = [S, C]
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
        E, S, C = [self.E] + self.params

        # FORWARD PASS
        # Embedding layer subspace
        self.z0    = T.ivector()                    # tweet in one hot

        # Use an intermediate sigmoid
        z1         = E[:, self.z0]                 # embedding
        z2         = T.nnet.sigmoid(T.dot(S, z1))  # subspace
        # Hidden layer
        z3         = T.dot(C, z2)
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
            param_list = [self.E.get_value()] + [W.get_value() 
                          for W in self.params] + [self.emb_path]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)
