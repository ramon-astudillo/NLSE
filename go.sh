# rm data/pkl/*.pkl
# python code/extract.py -f 'semeval_train.txt' 'tweets_2013.txt' 'tweets_2014.txt' 'tweets_2015.txt'
# echo "getting embedding"
# python code/extract.py -e 'data/txt/str_skip_600.txt'
echo "training"
python code/train.py data/pkl/semeval_train.pkl models/my_model.pkl
echo "testing"
python code/test.py models/my_model.pkl data/pkl/tweets_2013.pkl data/pkl/tweets_2014.pkl data/pkl/tweets_2015.pkl