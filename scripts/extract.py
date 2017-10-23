#!/usr/bin/python
import cPickle
import argparse
import numpy as np
import re
import os
import sys
from collections import Counter
# Local
from nlse.nlse import init_W
from nlse.features import tokenize_corpus

# Constants
INDEX_NAME = 'wrd2idx.pkl'
COUNT_NAME = 'wrd_count.pkl'
PRE_RANDOMIZATION = False


def read_semeval(file_path):
    tweets = []
    with open(file_path) as fid:
        for line in fid.readlines():
            tweet = line.rstrip().split('\t')[-1]
            meta = line.rstrip().split('\t')[2]
            tweet = [word for word in tweet.split()]
            tweets.append((meta, tweet))
    return tweets


def write_semeval(file_path, tweets):
    with open(file_path, 'w') as fid:
        for tweet in tweets:
            tweet_str = " ".join(tweet[1])
            fid.write('id1\tid2\t%s\tNone\t%s\n' % (tweet[0], tweet_str))


def split_train_dev(data_x, data_y, seed, perc=0.8):
    '''
    Split train set into train and dev
    '''

    # RANDOM SEED
    rng = np.random.RandomState(seed)

    if PRE_RANDOMIZATION:
        idx = np.arange(len(data_x))
        np.random.shuffle(idx)
        data_x = [data_x[i] for i in idx]
        data_y = [data_y[i] for i in idx]

    # Separate into the different classes
    data_pos_x = [data_x[i] for i in range(len(data_y)) if data_y[i] == 0]
    data_neg_x = [data_x[i] for i in range(len(data_y)) if data_y[i] == 1]
    data_neu_x = [data_x[i] for i in range(len(data_y)) if data_y[i] == 2]
    data_pos_y = [data_y[i] for i in range(len(data_y)) if data_y[i] == 0]
    data_neg_y = [data_y[i] for i in range(len(data_y)) if data_y[i] == 1]
    data_neu_y = [data_y[i] for i in range(len(data_y)) if data_y[i] == 2]

    # Divide into train/dev mantaining observed class distribution
    L_train_pos = int(len(data_pos_x)*perc)
    L_train_neg = int(len(data_neg_x)*perc)
    L_train_neu = int(len(data_neu_x)*perc)

    # Compose datasets
    train_x = (
        data_pos_x[:L_train_pos] +
        data_neg_x[:L_train_neg] +
        data_neu_x[:L_train_neu]
    )
    train_y = (
        data_pos_y[:L_train_pos] +
        data_neg_y[:L_train_neg] +
        data_neu_y[:L_train_neu]
    )
    dev_x = (
        data_pos_x[L_train_pos:] +
        data_neg_x[L_train_neg:] +
        data_neu_x[L_train_neu:]
    )
    dev_y = (
        data_pos_y[L_train_pos:] +
        data_neg_y[L_train_neg:] +
        data_neu_y[L_train_neu:]
    )
    # Shuffle them
    train_idx = np.arange(len(train_x))
    rng.shuffle(train_idx)
    train_x = [train_x[i] for i in train_idx]
    train_y = np.array([train_y[i] for i in train_idx])
    dev_idx = np.arange(len(dev_x))
    rng.shuffle(dev_idx)
    dev_x = [dev_x[i] for i in dev_idx]
    dev_y = np.array([dev_y[i] for i in dev_idx])

    # Inform user
    print "%d positive tweets\n%d negative tweets\n%d neutral tweets" % \
        (len(data_pos_x), len(data_neg_x), len(data_neu_x))
    N = len(data_pos_x) + len(data_neg_x) + len(data_neu_x)
    print "ratios %2.2f %2.2f %2.2f" % \
        (len(data_pos_x)*1./N, len(data_neg_x)*1./N, len(data_neu_x)*1./N)

    return train_x, train_y, dev_x, dev_y


def save_tokenized_corpus(in_fnames, text_folder, seed):
    '''
    Saves all files by separate but tokenized
    '''

    # RANDOM SEED
    # Optional storage of tokenized files
    in_fnames = args.f
    for in_fname in in_fnames:
        corpus = read_semeval(in_fname)
        corpus = tokenize_corpus(corpus)
        basename = os.path.basename(in_fname).split('.txt')[0]
        out_file = "%s/%s.tk.txt" % (text_folder, basename)
        write_semeval(out_file, corpus)
        print "%s -> %s" % (in_fname, out_file)
    print "Saved tokenized files under %s" % text_folder


def extract_feats(corpus, wrd2idx, one_hot):
    '''
    Convert semeval corpus into binary format
    '''
    # Extract data into matrix, take into account max size
    X = []
    y = []

    n_in = 0
    n_out = 0

    for tweet in corpus:
        # ONE-HOT WORD FEATURES
        tmp_x = []
        for wrd in tweet[1]:
            if wrd in wrd2idx:
                tmp_x.append(wrd2idx[wrd])
                n_in += 1
            else:
                # UNKNOWN
                tmp_x.append(1)
                n_out += 1
        # X.append(tmp_x)
        X.append(np.array(tmp_x).astype('int32'))
        # TARGETS
        if tweet[0] == 'positive':
            y.append(0)
        elif tweet[0] == 'negative':
            y.append(1)
        elif tweet[0] == 'neutral':
            y.append(2)
        else:
            raise ValueError("Unexpected Label! %s" % tweet[0])

    if one_hot:
        X = get_onehot(len(wrd2idx), X)

    return np.array(X), np.array(y)


def get_onehot(vocab_size, dataset):
        X = np.zeros((vocab_size, len(dataset)))
        for i, x in enumerate(dataset):
            X[x, i] = 1
        return X


def save_features(in_fnames, out_folder, no_tokenize=False, one_hot=False,
                  seed=1234):

    # READ AND TOKENIZE ALL CORPORA
    datasets = []
    for in_fname in in_fnames:
        corpus = read_semeval(in_fname)
        if not no_tokenize:
            corpus = tokenize_corpus(corpus)
        datasets.append(corpus)

    # BUILD INDEX FROM ALL CORPORA, ALSO STORE COUNTS
    # TODO: Change to a more compact form usint set()
    wrd2idx = {}
    idx = 0
    for d in datasets:
        for tweet in d:
            for wrd in tweet[1]:
                if wrd not in wrd2idx:
                    wrd2idx[wrd] = idx
                    idx += 1
    # Save index
    index_path = out_folder + '/' + INDEX_NAME
    with open(index_path, "w") as fid:
        print "Saving vocabulary %s" % (index_path)
        cPickle.dump(wrd2idx, fid, cPickle.HIGHEST_PROTOCOL)
    # Extra count for train data
    wrd_count = Counter()
    for tweet in datasets[0]:
        for wrd in tweet:
            wrd_count.update(wrd)
    # Save count
    count_path = out_folder + '/' + COUNT_NAME
    with open(count_path, "w") as fid:
        print "Saving word count %s" % (count_path)
        cPickle.dump(wrd_count, fid, cPickle.HIGHEST_PROTOCOL)

    # EXTRACT FEATURES FROM TRAIN DATA
    # This assumes that the first file refers to the training data!
    # Let's do a soft check, just in case
    assert re.match('.*train.*', in_fnames[0]), \
        "Are you sure the first file is the train data-set?"
    train_raw_x, train_raw_y = extract_feats(
        datasets[0], wrd2idx, one_hot=False
    )
    # shuffle traininig data and split into train and dev
    train_x, train_y, dev_x, dev_y = split_train_dev(train_raw_x, train_raw_y,
                                                     seed, perc=0.8)
    if one_hot:
        train_x = get_onehot(len(wrd2idx), train_x)
        dev_x = get_onehot(len(wrd2idx), dev_x)
    # Concatenate train data into a single numpy array, keep start and end
    # indices
    lens = np.array([len(tr) for tr in train_x])
    st = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype(int)
    ed = st + lens
    x = np.zeros((ed[-1], 1))
    for i, ins_x in enumerate(train_x):
        x[st[i]:ed[i]] = ins_x[:, None]
    train_x = x
    train_y = train_y[:, None]     # Otherwise slices are scalars not Tensors
    # Save training/dev features
    out_file = out_folder + '/' + 'train.pkl'
    with open(out_file, "w") as fid:
        print "Extracting %s -> %s" % (in_fnames[0], out_file)
        cPickle.dump([train_x, train_y, st, ed], fid, cPickle.HIGHEST_PROTOCOL)
    out_file = out_folder + '/' + 'dev.pkl'
    with open(out_file, "w") as fid:
        print "Extracting %s -> %s" % (in_fnames[0], out_file)
        cPickle.dump([dev_x, dev_y], fid, cPickle.HIGHEST_PROTOCOL)

    # EXTRACT AND SAVE THE REST OF THE DATA
    for in_fname, dataset in zip(in_fnames[1:], datasets[1:]):
        x, y = extract_feats(dataset, wrd2idx, one_hot)
        out_name = os.path.basename(os.path.splitext(in_fname)[0])
        out_file = out_folder + '/' + out_name + '.pkl'
        # Save
        with open(out_file, "w") as fid:
            print "Extracting %s -> %s" % (in_fname, out_file)
            cPickle.dump([x, y], fid, cPickle.HIGHEST_PROTOCOL)


def save_embedding(emb_path, pretrained_emb, index_path, UNK=False):
    '''
    Save a matrix of pre-trained embeddings
    '''

    if not os.path.isfile(index_path):
        raise IOError(
            "Unable to find the word index file\nRun with -f option to create"
            " the index file"
        )
    else:
        with open(index_path, "r") as fid:
            wrd2idx = cPickle.load(fid)

    if not os.path.isfile(emb_path):
        raise IOError("Unable to find the word embeddings file %s" % emb_path)

    print "Extracting %s -> %s" % (emb_path, pretrained_emb)
    with open(emb_path) as fid:
        # Get emb size
        _, emb_size = fid.readline().split()
        # Initialization
        rng = np.random.RandomState(1234)
        E = init_W((int(emb_size), len(wrd2idx)), rng, init=0.001, shared=False)

        # Get embeddings for all words in vocabulary
        found_words = set()
        for line in fid.readlines():
            items = line.split()
            wrd = items[0]
            if wrd in wrd2idx:
                E[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)
                found_words |= set([wrd])

    # Get list of ooev
    ooev_words = set(wrd2idx.keys()) - found_words

    # Number of out of embedding vocabulary embeddings
    n_OOEV = len(ooev_words)
    perc = n_OOEV*100./len(wrd2idx)
    print ("%d/%d (%2.2f %%) words in vocabulary found no embedding (OOEV) "
           "and were set to zero" % (n_OOEV, len(wrd2idx), perc))

    # save the ooev list
    ooev_file = "%s/%s.ooev.pkl" % (
        os.path.dirname(pretrained_emb),
        os.path.basename(pretrained_emb).split('.')[0]
    )
    with open(ooev_file, 'w') as fid:
        cPickle.dump(ooev_words, fid, cPickle.HIGHEST_PROTOCOL)
    print "OOEV list stored under %s" % ooev_file

    # save the embeddings
    with open(pretrained_emb, 'w') as fid:
        cPickle.dump(E, fid, cPickle.HIGHEST_PROTOCOL)


def save_embedding_noooev(emb_path, pretrained_emb, index_path, char_emb_path,
                          mapping_path, UNK=False):
    '''
    Save a matrix of pre-trained embeddings
    '''

    if not os.path.isfile(index_path):
        raise IOError(
            "Unable to find the word index file\nRun with -f option to create "
            "the index file"
        )
    else:
        with open(index_path, "r") as fid:
            wrd2idx = cPickle.load(fid)
        count_path = os.path.dirname(index_path) + '/' + COUNT_NAME

    if not os.path.isfile(emb_path):
        raise IOError("Unable to find the word embeddings file %s" % emb_path)

    if "nl" in mapping_path:
        non_linear = True
        print "non linear mapping"
    else:
        non_linear = False
        print "linear mapping"

    with open(mapping_path, "r") as fid:
        em_wrd2idx, emb_mapper = cPickle.load(fid)

    with open(char_emb_path) as fid:
        _, emb_size = fid.readline().split()
        E_char = np.zeros((int(emb_size), len(wrd2idx)))
        # Get embeddings for all words in vocabulary
        for line in fid.readlines():
            items = line.split()
            wrd = items[0]
            if wrd in wrd2idx:
                E_char[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)

    print "Extracting %s -> %s" % (emb_path, pretrained_emb)
    with open(emb_path) as fid:
        # Get emb size
        _, emb_size = fid.readline().split()
        # Initialization
        E = np.zeros((int(emb_size), len(wrd2idx)))
        found_words = set()
        # Get embeddings for all words in vocabulary
        for line in fid.readlines():
            items = line.split()
            wrd = items[0]
            if wrd in wrd2idx:
                E[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)
                found_words |= set([wrd])

    ooev_words = set(wrd2idx.keys()) - found_words
    # Get list of ooev
    ooev_words_idx = np.where(~E.any(axis=0))[0]
    for w in ooev_words_idx:
        char_emb = E_char[:, w]
        ssg_emb = np.dot(emb_mapper, char_emb)
        if non_linear:
            E[:, w] = np.tanh(ssg_emb)
        else:
            E[:, w] = ssg_emb

    # Number of out of embedding vocabulary embeddings
    n_OOEV = len(ooev_words)
    perc = n_OOEV*100./len(wrd2idx)
    print ("%d/%d (%2.2f %%) words in vocabulary found no embedding (OOEV) "
           "and were set to zero" % (n_OOEV, len(wrd2idx), perc))

    # save the ooev list
    ooev_file = "%s/%s.ooev.pkl" % (
        os.path.dirname(pretrained_emb),
        os.path.basename(pretrained_emb).split('.')[0]
    )
    with open(ooev_file, 'w') as fid:
        cPickle.dump(ooev_words, fid, cPickle.HIGHEST_PROTOCOL)
    print "OOEV list stored under %s" % ooev_file

    # save the embeddings
    with open(pretrained_emb, 'w') as fid:
        cPickle.dump(E, fid, cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(
        prog='Builds index from text and extracts embeddings')
    parser.add_argument(
        '-f', nargs='+',
        help='Train file and one or more test files in SemEval text format',
        type=str)
    parser.add_argument(
        '-e',
        help='Train file and one or more test files in SemEval text format',
        type=str)
    parser.add_argument(
        '-e2', nargs='+',
        help='Same as -e but using emebdding mapper',
        type=str)
    parser.add_argument(
        '-o',
        help=(
            'Folder where (-f) indexed dictionary, train/dev/test files or (-e)'
            ' the embeddings will be stored'
        ),
        type=str)
    parser.add_argument(
        '-t',
        help='Optional folder to store tokenized tweets',
        type=str
    )
    parser.add_argument(
        '-s',
        help='random seed for data shuffling, default is 1234',
        default=1234,
        type=int)
    parser.add_argument('--no-tokenize', action='store_true')

    args = parser.parse_args(sys.argv[1:])
    assert (bool(args.f) + bool(args.e) + bool(args.e2)) == 1, \
        "Either -f OR -e/-e2 options must be specified"

    if args.f:

        # Extract text files, create dictionary
        for in_fname in args.f:
            if not os.path.isfile(in_fname):
                raise EnvironmentError("Missing input file %s" % in_fname)

        if args.o:
            # Save features for files
            if not os.path.isdir(args.o):
                os.makedirs(args.o)
                print "Creating output folder %s" % args.o
            save_features(
                args.f, args.o, seed=args.s, no_tokenize=args.no_tokenize
            )

        if args.t:
            assert not args.no_tokenize, "tokenization must be used"
            save_tokenized_corpus(args.f, args.t, args.s)

    elif args.e or args.e2:

        # Extract embeddings
        index_path = '%s/%s' % (args.o, INDEX_NAME)
        if not os.path.isfile(index_path):
            raise EnvironmentError("Missing dictionary %s, did you run "
                                   "./code/extract.py -f ?\n" % index_path)
        if args.e:

            basename = os.path.basename(args.e).split('.txt')[0]
            pretrained_emb = '%s/%s.pkl' % (args.o, basename)
            if not os.path.isfile(args.e):
                raise EnvironmentError("Missing embeddings text file %s" %
                                       args.e)
            save_embedding(args.e, pretrained_emb, index_path)
        else:

            emb_path, char_emb_path, mapping_path = args.e2

            basename = os.path.basename(emb_path).split('.txt')[0]
            pretrained_emb = '%s/%s.pkl' % (args.o, basename)
            if not os.path.isfile(emb_path):
                raise EnvironmentError("Missing embeddings text file %s" %
                                       emb_path)
            save_embedding_noooev(emb_path, pretrained_emb, index_path,
                                  char_emb_path, mapping_path)
