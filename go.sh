#!/bin/bash -e

echo "Indexing data"
python code/extract.py -f DATA/txt/semeval_train.txt \
                          DATA/txt/tweets_2013.txt \
                          DATA/txt/tweets_2014.txt \
                          DATA/txt/tweets_2015.txt \
                          DATA/pkl/ 

echo "Getting embedding"
python code/extract.py -e DATA/txt/struct_skip_50.txt \
                          DATA/pkl/struct_skip_50.pkl \
                          DATA/pkl/wrd2idx.pkl
echo "Training"
python code/train.py data/pkl/semeval_train.pkl \
                     data/pkl/dev.pkl \
                     DATA/pkl/struct_skip_50.pkl \
                     DATA/models/sskip_50.pkl
echo "Testing"
python code/test.py DATA/models/sskip_50.pkl \
                    DATA/pkl/tweets_2013.pkl \
                    DATA/pkl/tweets_2014.pkl \
                    DATA/pkl/tweets_2015.pkl
