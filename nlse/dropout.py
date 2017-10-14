import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

def dropout(x, p, training=True, seed=1234):

    p = 1. - p

    srng = RandomStreams(seed)
    if training:
        x *= srng.binomial(size=x.shape, p=p, dtype=x.dtype)
        x /= p
        return x
    else:
        return x

if __name__ == '__main__':

    import numpy as np
    data_x = np.random.randn(1000, 1000).astype(theano.config.floatX)

    prob_droput = 0.0000001
    print "Testing dropout"
    x   = T.matrix()
    dbg = theano.function([x], dropout(x, prob_droput))
    print prob_droput
    print (dbg(data_x) == data_x).mean()
