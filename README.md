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

The code for these two papers is available in the `semeval2015` branch.
Current `master` points to the extended version for SemEval2016, for this cite,

    Amir, Silvio, Ramón Astudillo, Wang Ling, Mário J. Silva, and Isabel
    Trancoso. "INESC-ID at SemEval-2016 Task 4-A: Reducing the Problem of
    Out-of-Embedding Words." SemEval 2016.


## Instalation

The code is OLD, it uses Python2 and old theano. I reccomend a virtual
environment and upgrading to latest install tools

    virtualenv venv
    source venv/bin/activate
    pip install pip --upgrade
    pip install setuptools --upgrade

Then (or otherwise) just install in two steps

    pip install -r requirements.txt
    python setup.py develop

You will still need twokenize. Since there is no installer you will have to
download it and store it on the root folder of this repo, on unix just do

    wget https://raw.githubusercontent.com/myleott/ark-twokenize-py/master/twokenize.py

from the root folder of this repository. This should create the file

    twokenize.py

I assume that you will want to modify the code. Otherwise use

    python setup.py install

for a propper installation.

The go.sh bash script will need cygwin or equivalent in Windows machines, but
you can run the python commands inside the script on a Windows machine
directly. See the Step by Step section for instructions.

## Data

To reproduce the paper's results you will need the SemEval data from 2013 to
2016. This is not public so you have to ask someone for it (Silvio and I have
it). The paths for the data should be

    DATA/txt/Twitter2013_raws.txt
    DATA/txt/Twitter2014_raws.txt
    DATA/txt/Twitter2015_raws.txt
    DATA/txt/Twitter2016_raws.txt
    DATA/txt/semeval_train2016.txt

The data should look this example

    id1 id2 neutral None @USER i told you shane would get his 5th-star on rivals before signing day . @USER

You will also need to use some pre-trained embeddings. You can find the
embeddings we used here

    https://www.l2f.inesc-id.pt/~wlin/public/embeddings/

For example to download the 400 side emebddings (used by default in the go
script)

    wget https://www.l2f.inesc-id.pt/~wlin/public/embeddings/struc_skip_400.txt -P DATA/txt/ 

This should store the file as

    DATA/txt/struc_skip_400.txt

## Training your own embeddings

To train the embeddings with other data you can use

    https://github.com/wlin12/JNN

In case you want to use other embeddings the format is the following: The first
line should contain

    <number of words> <embedding size>

After that, the embedding for each word is specified as

    <word> <embedding_vector>

<embedding_vector> where each float is separated by a white space.

## Reproducing the Results

Once you have gotten the semeval data and the embeddings, just call

    ./go.sh

to extract the data and train the model. If you are using the largest embeddings
and you have no GPU, this might take a while. You can find more details about
the steps taken in the script
