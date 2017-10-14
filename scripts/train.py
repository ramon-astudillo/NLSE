#!/usr/bin/python
'''
Sub-space training
'''
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle
import argparse
import ast   # to pass literal values
import logging
import nlse.FMeasure as Fmes


def colstr(string, color):
    if color is None:
        cstring = string
    elif color == 'red':
        cstring = "\033[31m" + string + "\033[0m"
    elif color == 'green':
        cstring = "\033[32m" + string + "\033[0m"
    return cstring


if __name__ == '__main__':

    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(prog='Trains model')
    parser.add_argument(
        '-o',
        help='Folder where the train data embeddings are',
        type=str, required=True
    )
    parser.add_argument(
        '-e',
        help='Original embeddings file', type=str,
        required=True
    )
    parser.add_argument(
        '-m',
        help='Path where model is saved', type=str,
        required=True
    )
    # CONFIG
    # Model
    # TODO: Fuse all models and reduce this ton hidden layer type
    parser.add_argument(
        '-model',
        help='model used (MLP, GRU, CNN ...)',
        default="nlse", type=str
    )
    # Geometry
    parser.add_argument(
        '-sub_size',
        help='sub-space size', default=10,
        type=int
    )
    # Weight initialization
    parser.add_argument(
        '-normalize_embeddings',
        help='Normalize embeddings',
        default=False,
        type=ast.literal_eval
    )
    parser.add_argument(
        '-init_E_to_average',
        help='Initialize OOEV to the avearge', default=False,
        type=ast.literal_eval
    )
    parser.add_argument(
        '-s',
        help='random seed for data shuffling, default is 1234',
        default=1234,
        type=int
    )
    parser.add_argument(
        '-init_sub',
        help='Scale factor for sub-space initial uniform sampled weights',
        default=0.1
    )
    parser.add_argument(
        '-init_clas',
        help='Scale factor for classifier uniform sampled weights',
        default=0.7
    )
    # Optimization
    parser.add_argument(
        '-n_epoch',
        help='number of training epochs',
        default=12,
        type=int
    )
    parser.add_argument(
        '-lrate',
        help='learning rate',
        default=0.005,
        type=float
    )
    parser.add_argument(
        '-randomize',
        help='randomize each epoch',
        default=True,
        type=ast.literal_eval
    )
    parser.add_argument('-dropout', help='Dropout rate', default=0,
                        type=float)
    # Cost fuction
    parser.add_argument(
        '-neutral_penalty',
        help='Penalty for neutral cost',
        default=0.25,
        type=float
    )
    # OOEV: Update embeddings
    parser.add_argument(
        '-only_ooev',
        help='Update only OOEV',
        default=True,
        type=ast.literal_eval
    )
    parser.add_argument(
        '-update_emb',
        help='Update embeddings',
        default=True,
        type=ast.literal_eval
    )
    parser.add_argument(
        '-update_emb_until_iter',
        help='Update embeddings until iteration',
        default=3,
        type=int
    )
    parser.add_argument(
        '-lrate_emb',
        help='learning rate for embeddings',
        default=0.001,
        type=float
    )
    # Parse
    args = parser.parse_args(sys.argv[1:])
    # Manual parse for multi-type args
    try:
        args.init_sub = float(args.init_sub)
    except ValueError:
        args.init_sub = str(args.init_sub)
    try:
        args.init_clas = float(args.init_clas)
    except ValueError:
        args.init_clas = str(args.init_clas)

    # Model path
    dir_path = os.path.dirname(args.m)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # LOGGER
    basename = os.path.basename(args.m).split('.pkl')[0]
    log_path = '%s/%s.log' % (dir_path, basename)
    logging.basicConfig(level=logging.DEBUG,
                        filename=log_path,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    print "Log will be stored in %s" % log_path

    # Store config in log
    logging.debug('Config')
    for arg, value in vars(args).items():
        default_value = parser.get_default(arg)
        if default_value != value:
            logging.debug("\t%s = \033[34m%s\033[0m" % (arg, value))
        else:
            logging.debug("\t%s = %s" % (arg, value))

    # Get embedding paths
    basename = os.path.basename(args.e).split('.txt')[0]
    pretrained_emb = '%s/%s.pkl' % (args.o, basename)
    if not os.path.isfile(pretrained_emb):
        raise EnvironmentError("Missing extracted embeddings %s" %
                               pretrained_emb)
    # Random seed for epoch shuffle and embedding init
    rng = np.random.RandomState(args.s)

    # Weighted confusion matrix cost
    if args.neutral_penalty != 1:
        weigthed_CM = np.zeros((3, 3))
        weigthed_CM[0, :] = np.array([1, 0,                    0])  # positive
        weigthed_CM[1, :] = np.array([0, 1,                    0])  # negative
        weigthed_CM[2, :] = np.array([0, 0, args.neutral_penalty])  # neutral
        # Normalize
        weigthed_CM = weigthed_CM*3./weigthed_CM.sum()
        weigthed_CM = weigthed_CM.astype(theano.config.floatX)
        weigthed_CM = theano.shared(weigthed_CM, borrow=True)
    else:
        weigthed_CM = None

    # Create model
    if args.model == 'nlse':
        import nlse.nlse as model   # model used
        nn = model.NN(pretrained_emb,
                      args.sub_size,
                      dropout_prob=args.dropout,
                      init=args.s,
                      init_sub=args.init_sub,
                      init_clas=args.init_clas,
                      weight_CM=weigthed_CM)
    elif args.model == 'gru':
        import nlse.gruse as model   # model used
        nn = model.NN(pretrained_emb, args.sub_size, weight_CM=weigthed_CM,
                      init=args.s, dropout_prob=args.dropout)
    else:
        raise NotImplementedError("Model %s not supported" % args.model)
    logging.info('Initialized model from embeddings %s' % pretrained_emb)

    # SEMEVAL TRAIN TWEETS
    train_data = "%s/train.pkl" % args.o
    dev_data = "%s/dev.pkl" % args.o
    logging.info("Training data: %s" % train_data)
    logging.info("Dev data: %s" % dev_data)
    with open(train_data, 'rb') as fid:
        train_x, train_y, st, ed = cPickle.load(fid)
    with open(dev_data, 'rb') as fid:
        dev_x, dev_y = cPickle.load(fid)
    n_sent_train = len(st)
    # Ensure types compatible with GPU
    train_x = train_x.astype('int32')
    train_y = train_y.astype('int32')
    st = st.astype('int32')
    ed = ed.astype('int32')
    # Store as shared variables (push into the GPU)
    train_x = theano.shared(train_x, borrow=True)
    train_y = theano.shared(train_y, borrow=True)
    st = theano.shared(st, borrow=True)
    ed = theano.shared(ed, borrow=True)

    # SGD Update rule
    E = nn.params[0]
    # Sub-space: Do not update E
    updates = [(pr, pr-args.lrate*T.grad(nn.F, pr)) for pr in nn.params[1:]]

    # Normalize embeddigs per dimension
    if args.normalize_embeddings:
        E_val = E.get_value()
        mu = E_val.mean(1, keepdims=True)
        std = E_val.std(1, keepdims=True)
        E.set_value((E_val-mu)/std)

    # UPDATE ALSO EMBEDDINGS
    if args.update_emb:

        logging.info("Will update embeddings")

        # Create a mask that forces update of only OOEVs
        [emb_size, voc_size] = E.get_value().shape
        if args.only_ooev:

            # CHAPUZA: Hard-coded
            # Load list of OOEV indices
            with open('%s/wrd2idx.pkl' % args.o) as fid:
                word2idx = cPickle.load(fid)
            ooev_file = '%s/%s.ooev.pkl' % \
                (os.path.dirname(pretrained_emb),
                 os.path.basename(pretrained_emb).split('.')[0])
            with open(ooev_file) as fid:
                idx = np.array([word2idx[wr] for wr in cPickle.load(fid)])
            # Create mask
            mask = np.zeros((1, voc_size)).astype(theano.config.floatX)
            mask[0, idx] = 1.
            logging.info(
                "Only %d OOEV updated (if they appear in training)" %
                idx.shape[0]
            )
        else:
            # Update all embeddings
            mask = np.ones((1, voc_size)).astype(theano.config.floatX)
        mask = theano.shared(mask, borrow=True, broadcastable=(True, False))

        if args.init_E_to_average:
            idx = np.nonzero(E.get_value().sum(0) == 0)[0]
            not_idx = np.nonzero(E.get_value().sum(0) != 0)[0]
            E_val = E.get_value()
            mu = E_val[:, not_idx].mean(1, keepdims=True)
            std = E_val[:, not_idx].std(1, keepdims=True)
            embs, vocs = E_val[:, idx].shape
            E_val[:, idx] = std*rng.randn(embs, vocs) + mu
            E.set_value(E_val)

        # Sparse update
        sp_grad = T.grad(nn.F, nn.z1)
        inc = T.inc_subtensor(nn.z1, -args.lrate_emb*mask[0, nn.z0]*sp_grad)
        updates += [(E, inc)]

    # Batch
    i = T.lscalar()
    givens = {nn.z0: train_x[st[i]:ed[i], 0], nn.y: train_y[i]}
    # Compile
    train_batch = theano.function([i], nn.F, updates=updates, givens=givens)
    train_idx = np.arange(n_sent_train).astype('int32')

    # TRAIN
    last_obj = None
    last_Fm = None
    best_Fm = [0, 0]
    last_Acc = None
    stop = False
    for i in np.arange(args.n_epoch):

        # Do not update embeddings furthermore
        if args.update_emb and (i+1 >= args.update_emb_until_iter):
            if not stop:
                logging.info("Stopped updating embeddings")
                stop = True
            mask.set_value(np.zeros((1, voc_size)).astype(theano.config.floatX))

        # Epoch train
        obj = 0
        n = 0
        if args.randomize:
            rng.shuffle(train_idx)
        for j in train_idx:
            obj += train_batch(j)
            # INFO
            if not (n % 500):
                print "\rTraining %d/%d" % (n+1, n_sent_train),
                sys.stdout.flush()
            n += 1

        # Evaluation
        cr = 0.
        mapp = np.array([1, 2, 0])
        ConfMat = np.zeros((3, 3))
        dev_hat_y = np.zeros(dev_y.shape, dtype='int32')
        dev_p_y = np.zeros((3, dev_y.shape[0]))
        for j, x, y in zip(np.arange(len(dev_x)), dev_x, dev_y):
            # Prediction
            p_y = nn.forward(x)
            hat_y = np.argmax(p_y)
            # Confusion matrix
            ConfMat[mapp[y], mapp[hat_y]] += 1
            # Accuracy
            cr = (cr*j + (hat_y == y).astype(float))/(j+1)
            # INFO
            if not (j % 500):
                print "\rTraining %d/%d Devel %d/%d" % \
                    (n, n_sent_train, j+1, len(dev_x)),
                sys.stdout.flush()
        print "\rTraining %d/%d Devel %d/%d\n" % \
            (n, n_sent_train, j+1, len(dev_x)),

        # Compute SemEval scores
        Fm = Fmes.FmesSemEval(confusionMatrix=ConfMat)

        # INFO
        if last_Fm:
            if best_Fm[0] < Fm:
                # Keep best model
                best_Fm = [Fm, i+1]
                nn.save(args.m)
                best = '*'
            else:
                best = ''
            delta_Fm = Fm - last_Fm
            if delta_Fm >= 0:
                delta_str = colstr("+%2.2f" % (delta_Fm*100), 'green')
            else:
                delta_str = colstr("%2.2f" % (delta_Fm*100), 'red')
            if obj < last_obj:
                obj_str = colstr("%e" % obj, 'green')
            else:
                obj_str = colstr("%e" % obj, 'red')
            last_obj = obj

        else:

            # First model is best model
            best_Fm = [Fm, i+1]
            obj_str = "%e" % obj
            last_obj = obj
            delta_str = ""
            best = ""
            nn.save(args.m)

        if last_Acc:
            if last_Acc > cr:
                acc_str = "Acc " + colstr("%2.2f%%" % (cr*100), 'red')
            else:
                acc_str = "Acc " + colstr("%2.2f%%" % (cr*100), 'green')
        else:
            acc_str = "Acc %2.2f%%" % (cr*100)
        last_Acc = cr
        last_Fm = Fm
        items = (i+1, args.n_epoch, obj_str, acc_str, Fm*100, delta_str, best)
        logging.info("Epoch %2d/%2d: %s %s Fm %2.2f%% %s%s" % items)
