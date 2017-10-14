from theano.tensor.shared_randomstreams import RandomStreams


def dropout(x, p, training=True, seed=1234):
    p = 1. - p
    srng = RandomStreams(seed)
    if training:
        x *= srng.binomial(size=x.shape, p=p, dtype=x.dtype)
        x /= p
        return x
    else:
        return x
