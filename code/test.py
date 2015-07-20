import cPickle
import FMeasure as Fm
import nlse
import numpy as np
import sys
from ipdb import set_trace
# Pre-trained embeddings
pretrained_emb = 'data/pkl/Emb.pkl'

def main(model_path, test_sets):

	nn = nlse.NLSE(pretrained_emb, model_file=model_path)

	for test_set in test_sets:
		print "Testing dataset: %s" % test_set
		with open(test_set, 'rb') as fid:
			X, Y = cPickle.load(fid) 
			X = X[:20]
			Y = Y[:20]			

			acc     = 0.
			mapp    = np.array([ 1, 2, 0])
			conf_mat = np.zeros((3, 3))
			for j, x, y in zip(np.arange(len(X)), X, Y):
				# Prediction
				p_y   = nn.forward(x)
				hat_y = np.argmax(p_y)
				# Confusion matrix
				conf_mat[mapp[y], mapp[hat_y]] += 1
				# Accuracy
				acc    = (acc*j + (hat_y == y).astype(float))/(j+1)
				if not (j % 5):
					sys.stdout.write("\rTesting %d/%d" % (j+1, len(X)))
					sys.stdout.flush()   
			fm = Fm.FmesSemEval(confusionMatrix=conf_mat)
			# set_trace()

			print "\n\nAvg. F-measure (POS/NEG): %.3f" % fm
			print "\nAccuracy: %.3f" % acc

def check_args(args):
	pass

if __name__ == '__main__':
    
    check_args(sys.argv)
    model_path = sys.argv[1]
    test_sets = sys.argv[2:]
    
    main(model_path, test_sets)