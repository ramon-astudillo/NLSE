# NLSE
Non-Linear Sub-Space Embedding model

Tokenization (./twokenize.py)

Will store as DATA/pkl/<name>.pkl and create a voc. 
./code/extract.py DATA/txt/semeval_train.txt \
                  DATA/txt/semeval2013.txt \
                  DATA/txt/semeval2014.txt \
                  DATA/txt/semeval2015.txt
                  
Selects embeddings for the voc.

Train
./code/train.py DATA/pkl/semeval_train.pkl DATA/models/mymodel.pkl

Get SemEval results
./code/test.py DATA/pkl/semeval2013.pkl \
               DATA/pkl/semeval2014.pkl \
               DATA/pkl/semeval2015.pkl
               
  
                  

