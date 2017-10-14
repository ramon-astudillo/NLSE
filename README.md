NLSE
====
Non-Linear Sub-Space Embedding model used for the SemEval challenge described
in

    R. F. Astudillo, S. Amir,  W. Ling, B. Martins, M. Silva and I. Trancoso "INESC-ID:
    Sentiment Analysis without hand-coded Features or Liguistic Resources using Embedding
    Subspaces", SemEval 2015

[[pdf]](http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval109.pdf), [[BibTex]](https://scholar.google.pt/scholar.bib?q=info:ocLxCnCv3BIJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAVdckz2Jdg1II6YtdC0iMIN9l2RyFix9R&scisf=4&hl=en)

For the extended experiments, including POS tagging, please cite

    R. F. Astudillo, S. Amir,  W. Ling, M. Silva and I. Trancoso "Learning Word Representations
    from Scarce and Noisy Data with Embedding Sub-spaces", ACL-IJCNLP 2015

[[pdf]](http://anthology.aclweb.org/P/P15/P15-1104.pdf),
[[BibTex]](https://scholar.google.pt/scholar.bib?q=info:0rog_aWHY1QJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAVdclXBe81CgJ3lNDs6Y5Ul2Zjrbi7nxu&scisf=4&hl=en)

## Instalation

The code is OLD, it uses Python2 and old theano (pytorch is the way now). I
reccomend a virtual environment

    virtualenv venv
    pip install pip --upgrade
    pip install -r requirements.txt
    python setup.py develop

Yo will still need twokenize. Since there is no installer you will have to
download it and store it on the root folder of this repo, on unix just do

    wget https://raw.githubusercontent.com/myleott/ark-twokenize-py/master/twokenize.py

I assume that you will want to modify the code. Otherwise use

    python setup.py install

for a propper installation.

The go.sh bash script will need cygwin or equivalent in Windows machines, but
you can run the python commands inside the script on a Windows machine
directly. See the Step by Step section for instructions.

## Data

To reproduce the paper's results you will need the SemEval data from 2013 to
2015. You also need to tokenize the data using

    https://github.com/myleott/ark-twokenize-py

If done right, each tweet should occupy one line, and should look like this example

    id1 id2 neutral None @USER i told you shane would get his 5th-star on rivals before signing day . @USER

You will also need to use some pre-trained embeddings. You can find the
embeddings we used here

    https://www.l2f.inesc-id.pt/~wlin/public/embeddings/

to train the embeddings with other data you can use

    https://github.com/wlin12/JNN

In case you want to use your own the first line should contain

    <number of words> <embedding size>

After that, the embedding for each word is specified as

    <word> <embedding_vector>

<embedding_vector> where each float is separated by a white space.

## Reproducing the Results

Once you have formatted the data and obtained the embeddings, you just have to run

    ./go.sh

to extract the data and train the model. If you are using the largest embeddings
and you have no GPU, this might take a while.

## Step by Step Explanation

In case, you want to use this code with other data-sets here is a detailed
description of what you need to do. These are also the steps followed inside
go.sh

First you create the index and global vocabulary from the text-based data. For
this you need to use

    python code/extract.py -f DATA/txt/semeval_train.txt \
                              DATA/txt/tweets_2013.txt \
                              DATA/txt/tweets_2014.txt \
                              DATA/txt/tweets_2015.txt \
                              DATA/pkl/

This will store Pickle files with same file name as the txt files under

    DATA/pkl/

It will also store a wrd2idx.pkl containing a dictionary that maps each word to
an integer index. If you have any number of txt files using the same format,
it should work as well.

Next thing is to select the pre-trained embeddings present in the vocabulary
you are using to build your embedding matrix. This is done with

    python code/extract.py -e DATA/txt/struct_skip_50.txt \
                              DATA/pkl/struct_skip_50.pkl \
                              DATA/pkl/wrd2idx.pkl

Note that this step can be a source of problems. If you have words that are not
in your embeddings they will be set to an embedding of zero. This can be
counter-productive in some cases.

To train the model use

    python code/train.py DATA/pkl/semeval_train.pkl \
                         DATA/pkl/dev.pkl \
                         DATA/pkl/struct_skip_50.pkl \
                         DATA/models/sskip_50.pkl

Here

    DATA/models/sskip_50.pkl

Is just a example name to define the model. You should use more detailed names
to remember the hyper-parameters used.

Finally to get the SemEval results, you just need to do

    python code/test.py DATA/models/sskip_50.pkl \
                        DATA/pkl/tweets_2013.pkl \
                        DATA/pkl/tweets_2014.pkl \
                        DATA/pkl/tweets_2015.pkl
