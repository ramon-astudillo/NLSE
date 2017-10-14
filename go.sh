#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# DATA-SETS
train_set="DATA/txt/semeval_train2016.txt"
test_sets="DATA/txt/Twitter2013_raws.txt 
           DATA/txt/Twitter2014_raws.txt 
           DATA/txt/Twitter2015_raws.txt" 

embeddings=DATA/txt/struc_skip_50.txt
#embeddings=DATA/txt/str_skip_400.txt

# Model 
model=nlse

# Geometry
sub_size=100     # 10 

# Weight initialization
init_sub='glorot-sigmoid'    # 0.1
init_clas='glorot-sigmoid'   # 0.7

# Optimization    
n_epoch=100     # 12
lrate=0.0001    # 0.005
randomize=True  # True    
dropout=0.      # 0.

# Cost fuction    
neutral_penalty=0.25 # 0.25 
update_emb=True      # True

work_folder=DATA/pkl/$(basename $train_set .txt)/
model_path=$work_folder/models/$(basename $embeddings .txt)/sub_size${sub_size}.$dropout/
model_name=semeval2016.pkl

# OTHER
if [ ! -d "$model_path" ];then
    mkdir -p "$model_path"
fi

printf "\033[34mIndexing all data\033[0m\n"
python scripts/extract.py -o $work_folder -f $train_set $test_sets 

printf "\033[34mExtracting embeddings for all data\033[0m\n"
python scripts/extract.py -o $work_folder -e $embeddings

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

printf "\033[34mTesting\033[0m\n"
python scripts/test.py -o $work_folder \
                    -m $model_path/$model_name \
                    -f $test_sets -model nlse 
