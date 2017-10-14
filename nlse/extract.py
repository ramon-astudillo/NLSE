#!/usr/bin/python
import cPickle
from ipdb import set_trace
import numpy as np
import os
import tarfile
import sys

INDEX_NAME = 'wrd2idx.pkl'  

def help():
    print "\nTo index tokenized text files (The first must be the train-set!)\n\npython code/extract.py -f train.txt test_1.txt [ ... test_2.txt] pkl_folder/\n\nTo extract embeddings for this vocabulary (You need the pkl_folder/index.pkl created with the previous command):\n\npython code/extract.py -e embeddings.txt embeddings.pkl pkl_folder/index.pkl\n"
    exit(1)

def split_train_dev(train_x, train_y, perc=0.8):
    '''
    Split train set into train and dev
    '''

    # RANDOM SEED
    rng = np.random.RandomState(1234)

    # Ensure data is suitable for theano
    data_x = train_x #[tx.astype('int32') for tx in train_x]
    data_y = train_y #[np.array(ty).astype('int32')[None] for ty in train_y]
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
    train_x = (data_pos_x[:L_train_pos] + data_neg_x[:L_train_neg] 
               + data_neu_x[:L_train_neu])
    train_y = (data_pos_y[:L_train_pos] + data_neg_y[:L_train_neg] 
               + data_neu_y[:L_train_neu])
    dev_x   = (data_pos_x[L_train_pos:] + data_neg_x[L_train_neg:] 
               + data_neu_x[L_train_neu:])
    dev_y   = (data_pos_y[L_train_pos:] + data_neg_y[L_train_neg:] 
               + data_neu_y[L_train_neu:])
    # Shuffle them
    train_idx = np.arange(len(train_x))
    rng.shuffle(train_idx)
    train_x   = [train_x[i] for i in train_idx]
    train_y   = np.array([train_y[i] for i in train_idx])
    dev_idx   = np.arange(len(dev_x))
    rng.shuffle(dev_idx)
    dev_x     = [dev_x[i] for i in dev_idx]
    dev_y     = np.array([dev_y[i] for i in dev_idx])
    
    return train_x, train_y, dev_x, dev_y 

def extract_feats(corpus, wrd2idx, one_hot):
    '''
    Convert semeval corpus into binary format    
    '''
    # Extract data into matrix, take into account max size
    X = [] 
    y = []
    
    n_in  = 0
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
            raise ValueError, ("Unexpected Label! %s" % tweet[0])
        
    if one_hot:        
        X = get_onehot(len(wrd2idx),X)

    return np.array(X), np.array(y)

def read_corpus(corpus_path):

    with open(corpus_path) as f:
        corpus = [(line.split()[2], line.split()[4:]) for line in f.readlines()]

    return corpus

def get_onehot(vocab_size, dataset):
        
        X = np.zeros((vocab_size,len(dataset)))
        for i, x in enumerate(dataset):
            X[x,i] = 1
            
        return X

def save_features(in_fnames, out_folder, one_hot=False):

    # READ ALL CORPORA 
    datasets = [read_corpus(in_fname) for in_fname in in_fnames]

    # BUILD INDEX FROM ALL CORPORA
    wrd2idx = {}
    idx = 0        
    for d in datasets:
        for tweet in d:
            for wrd in tweet[1]:
                if wrd not in wrd2idx:
                    wrd2idx[wrd] = idx
                    idx += 1       

    # SAVE INDEX
    index_path = out_folder + '/' + INDEX_NAME
    with open(index_path,"w") as fid:
        print "Saving vocabulary %s" % (index_path)
        cPickle.dump(wrd2idx, fid, cPickle.HIGHEST_PROTOCOL)
    
    #EXTRACT FEATURES
    #ASSUMES THAT THE FIRST FILE REFERS TO THE TRAINING DATA        
    train_raw_x, train_raw_y = extract_feats(datasets[0], wrd2idx, one_hot=False) 
    #shuffle traininig data and split into train and dev
    train_x, train_y, dev_x, dev_y = split_train_dev(train_raw_x, train_raw_y, 
                                                     perc=0.8)
    
    if one_hot:
        train_x = get_onehot(len(wrd2idx), train_x)
        dev_x   = get_onehot(len(wrd2idx), dev_x)        
    #save training/dev features        
    out_name = os.path.basename(os.path.splitext(in_fnames[0])[0])
    out_file = out_folder + '/' + out_name + '.pkl'
    with open(out_file,"w") as fid:
        print "Extracting %s -> %s" % (in_fnames[0], out_file)  
        cPickle.dump([train_x, train_y], fid, cPickle.HIGHEST_PROTOCOL)

    out_file = out_folder + '/' + 'dev.pkl'
    with open(out_file,"w") as fid:
        print "Extracting %s -> %s" % (in_fnames[0], out_file)  
        cPickle.dump([dev_x, dev_y], fid, cPickle.HIGHEST_PROTOCOL)

    for in_fname, dataset in zip(in_fnames[1:], datasets[1:]):
        x, y     = extract_feats(dataset, wrd2idx, one_hot)
        out_name = os.path.basename(os.path.splitext(in_fname)[0])
        out_file = out_folder + '/' + out_name + '.pkl'
        with open(out_file, "w") as fid:
            print "Extracting %s -> %s" % (in_fname, out_file)  
            cPickle.dump([x, y], fid, cPickle.HIGHEST_PROTOCOL)

def save_embedding(emb_path, pretrained_emb, index_path):
    
    '''
        Save a matrix of pre-trained embeddings
    '''

    if not os.path.isfile(index_path):
        raise IOError, ("Unable to find the word index file\nRun with -f" 
                        "option to create the index file")
    else:
        with open(index_path,"r") as fid:
            wrd2idx = cPickle.load(fid)

    if not os.path.isfile(emb_path):
        raise IOError, ("Unable to find the word embeddings file %s" % emb_path)
    
    print "Extracting %s -> %s" % (emb_path, pretrained_emb)  

    with open(emb_path) as fid:        
        # Get emb size
        _, emb_size = fid.readline().split()
        # Get embeddings for all words in vocabulary
        E = np.zeros((int(emb_size), len(wrd2idx)))   
        for line in fid.readlines():                    
            items = line.split()
            wrd   = items[0]
            if wrd in wrd2idx:
                E[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)
        # Number of out of embedding vocabulary embeddings
        n_OOEV = np.sum((E.sum(0) == 0).astype(int))
        perc = n_OOEV*100./len(wrd2idx)
        print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
               % (n_OOEV, len(wrd2idx), perc)) 
    #save the embeddings
    with open(pretrained_emb, 'w') as fid:
        cPickle.dump(E, fid, cPickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":    

    # ARGUMENT HANDLING 
    if len(sys.argv[1:]) > 3:
        opt = sys.argv[1]
        if opt == "-f":        

            # This should be the input text files and the output folder
            in_fnames  = sys.argv[2:-1]   
            out_folder = sys.argv[-1]
            # SANITY CHECKS
            for in_fname in in_fnames:
                if not os.path.isfile(in_fname):
                    print "\nMissing input file %s\n" % in_fname
                    help()
            if not os.path.isdir(out_folder):        
               os.mkdir(out_folder)
               print "Creating output folder %s" % out_folder 
            print ""   
            save_features(in_fnames, out_folder)
            print ""   

        elif opt == "-e":

            # This should be the input full embeddings in text form, output
            # embeddings reduced to a vocabulary and the vocabulary
            emb_path, pretrained_emb, index_path = sys.argv[2:]            
            # SANITY CHECKS
            if not os.path.isfile(emb_path):
                print "\nMissing embeddings text file %s\n" % emb_path
                help()
            if not os.path.isfile(index_path):
                print ("\nMissing dictionary %s, did you run "
                       "./code/extract.py -f ?\n" % index_path)
                help()
            print ""   
            save_embedding(emb_path, pretrained_emb, index_path)
            print ""   

    else:
        print "\nNot enough arguments!" 
        help()
