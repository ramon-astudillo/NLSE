#!/bin/bash -e

printf "\nTODO: Tokenize data\n"

echo "Indexing all data"
python code/extract.py -f DATA/txt/semeval_train.txt \
                          DATA/txt/tweets_2013.txt \
                          DATA/txt/tweets_2014.txt \
                          DATA/txt/tweets_2015.txt \
                          DATA/pkl/ 

echo "Extrating embeddings for all data"
python code/extract.py -e DATA/txt/struc_skip_600.txt \
                          DATA/pkl/struc_skip_600.pkl \
                          DATA/pkl/wrd2idx.pkl

echo "Training"
python code/train.py DATA/pkl/semeval_train.pkl \
                     DATA/pkl/dev.pkl \
                     DATA/pkl/struc_skip_600.pkl \
                     DATA/models/SemEval_struc_skip_600.pkl

echo "Testing"
python code/test.py DATA/models/SemEval_struc_skip_600.pkl \
                    DATA/pkl/tweets_2013.pkl \
                    DATA/pkl/tweets_2014.pkl \
                    DATA/pkl/tweets_2015.pkl
