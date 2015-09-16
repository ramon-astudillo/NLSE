import cPickle
import FMeasure as Fmes
import nlse
import numpy as np
import sys
from ipdb import set_trace
# Pre-trained embeddings
pretrained_emb = 'data/pkl/Emb.pkl'


def help():
    print "\npython code/test.py model_path test_file_1 ... test_file_n\n"
    exit()
            
if __name__ == '__main__':

    model_path = sys.argv[1]
    test_sets  = sys.argv[2:]

    # LOAD MODEL
    # NOTE: The model file already contains a path to the pre-trained
    # embeddings used
    nn = nlse.NLSE(None, None, model_file=model_path)

    # TEST ON EACH TEST SET
    for test_set in test_sets:
        print "\nTesting dataset: %s" % test_set
        with open(test_set, 'rb') as fid:
            X, Y     = cPickle.load(fid)             
            acc      = 0.
            mapp     = np.array([ 1, 2, 0])
            conf_mat = np.zeros((3, 3))
            for j, x, y in zip(np.arange(len(X)), X, Y):
                # Prediction
                p_y   = nn.forward(x)
                hat_y = np.argmax(p_y)
                # Confusion matrix
                conf_mat[mapp[y], mapp[hat_y]] += 1
                # Accuracy
                acc    = (acc*j + (hat_y == y).astype(float))/(j+1)
                if not (j % 100):
                    sys.stdout.write("\rTesting %d/%d" % (j+1, len(X)))
                    sys.stdout.flush()   
            fm = Fmes.FmesSemEval(confusionMatrix=conf_mat)
            # set_trace()

            sys.stdout.write("\rAcc: %2.2f%% | Fm: %2.2f%%\n" % ( acc*100, fm*100))
            sys.stdout.flush()   
