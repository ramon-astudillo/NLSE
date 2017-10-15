#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# DATA-SETS
train_set="DATA/txt/semeval_train2016.txt"
test_sets="DATA/txt/Twitter2013_raws.txt
           DATA/txt/Twitter2014_raws.txt
           DATA/txt/Twitter2015_raws.txt
           DATA/txt/Twitter2016_raws.txt"

embeddings=DATA/txt/struc_skip_400.txt

# Model
model=nlse

# Geometry
sub_size=10

# Weight initialization
init_sub=0.1
init_clas=0.7

# Optimization
n_epoch=12
lrate=0.005
randomize=True
dropout=0.

# Cost fuction
neutral_penalty=0.25
update_emb=True

work_folder=DATA/pkl/$(basename $train_set .txt)/
model_path=$work_folder/models/$(basename $embeddings .txt)/sub_size${sub_size}.$dropout/
model_name=semeval2016.pkl

# OTHER
if [ ! -d "$model_path" ];then
    mkdir -p "$model_path"
fi

# First you create the index and global vocabulary from the text-based data.
# This will store Pickle files with same file name as the txt files under
#
#    DATA/pkl/
#
# It will also store a wrd2idx.pkl containing a dictionary that maps each word
# to an integer index. If you have any number of txt files using the same
# format, it should work as well.
printf "\033[34mIndexing all data\033[0m\n"
python scripts/extract.py -o $work_folder -f $train_set $test_sets

# Next thing is to select the pre-trained embeddings present in the vocabulary
# you are using to build your embedding matrix. This is done with
printf "\033[34mExtracting embeddings for all data\033[0m\n"
python scripts/extract.py -o $work_folder -e $embeddings

# Note that this step can be a source of problems. If you have words that are
# not in your embeddings they will be set to an embedding of zero. This can be
# counter-productive in some cases.
printf "\033[34mTraining\033[0m\n"
python scripts/train.py -o $work_folder -e $embeddings \
                     -m $model_path/$model_name \
                     -model $model \
                     -lrate $lrate \
                     -n_epoch $n_epoch   \
                     -neutral_penalty $neutral_penalty \
                     -randomize $randomize \
                     -update_emb $update_emb \
                     -sub_size $sub_size \
                     -init_sub $init_sub \
                     -init_clas $init_clas \
                     -dropout $dropout

# Finally to get the SemEval results, you just need to do
printf "\033[34mTesting\033[0m\n"
python scripts/test.py -o $work_folder \
                    -m $model_path/$model_name \
                    -f $test_sets -model nlse
