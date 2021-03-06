NLSE
====
Non-Linear Sub-Space Embedding model used for the SemEval 2015 challenge
described in

    R. F. Astudillo, S. Amir,  W. Ling, B. Martins, M. Silva and I. Trancoso "INESC-ID:
    Sentiment Analysis without hand-coded Features or Liguistic Resources using Embedding
    Subspaces", SemEval 2015

[[pdf]](http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval109.pdf)

For the extended experiments, including POS tagging, please cite

    R. F. Astudillo, S. Amir,  W. Ling, M. Silva and I. Trancoso "Learning Word Representations
    from Scarce and Noisy Data with Embedding Sub-spaces", ACL-IJCNLP 2015

[[pdf]](http://anthology.aclweb.org/P/P15/P15-1104.pdf)

The code for these two papers is available in the `semeval2015` branch.
Current `master` points to the extended version for SemEval 2016, for this
cite,

    Amir, Silvio, Ramón Astudillo, Wang Ling, Mário J. Silva, and Isabel Trancoso. "INESC-ID at 
    SemEval-2016 Task 4-A: Reducing the Problem of Out-of-Embedding Words." SemEval 2016.

[[pdf]](http://www.aclweb.org/anthology/S16-1036) 


## Instalation

The code uses Python2 and theano. I recommend a virtual environment and
upgrading to latest install tools

    virtualenv venv
    source venv/bin/activate
    pip install pip --upgrade
    pip install setuptools --upgrade

Then (or otherwise) just install in two steps

    pip install -r requirements.txt
    python setup.py develop

You will also need twokenize. Since there is no installer you will have to
download it and store it on the root folder of this repo, on unix just do

    wget https://raw.githubusercontent.com/myleott/ark-twokenize-py/master/twokenize.py

from the root folder of this repository. This should create the file

    twokenize.py

The go.sh bash script will need cygwin or equivalent in Windows machines, but
you can run the python commands inside the script on a Windows machine
directly. See inside the script for step by step details. 

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

For example to download the 400 side embeddings (used by default in the go
script)

    wget https://www.l2f.inesc-id.pt/~wlin/public/embeddings/struc_skip_400.txt -P DATA/txt/ 

This should store the file as

    DATA/txt/struc_skip_400.txt

## Reproducing the Results

Once you installed and obtained the data, just call

    ./go.sh

to extract the data and train the model. If you are using the largest embeddings
and you have no GPU, this might take a while. You can find more details about
the steps taken in the script.

If everything goes fine you should get following results, which are slightly
above those reported on the last paper.


| Test-Set   | Accuracy    | F-measure (sum)  |
| -----------|:-----------:| :-----:|
| 2013       | 72.82%      | 0.724 |
| 2014       | 73.87%      | 0.723 |
| 2015       | 68.19%      | 0.658 |
| 2016       | 61.36%      | 0.615 |

## Training your own embeddings

To train the embeddings with other data you can use

    https://github.com/wlin12/JNN

In case you want to use other embeddings the format is the following: The first
line should contain

    <number of words> <embedding size>

After that, the embedding for each word is specified as

    <word> <embedding_vector>

`<embedding_vector>` where each float is separated by a white space.

## Troubleshooting 

As you may know theano is no more longer supported. It was a great start for
Computational Graph toolkits but there are much better alternatives out there
nowadays (Pytorch, TensorFlow). If you have problems running it, try disabling
cuDNN. For this you need to write the following on your `~/.theanorc` 

    [dnn]
    enabled = False

If you are a newtimer to theano, to get it running on the GPU the faster way is
to define this on your `~/.bashrc`

    export THEANO_FLAGS='cuda.root=/usr/local/cuda-<version>/,device=gpu,floatX=float32'
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-<version>/lib64"

where `<version>` is the CUDA version.
