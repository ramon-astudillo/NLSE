import cPickle
import FMeasure as Fmes
import nlse
import numpy as np
import sys
from ipdb import set_trace
# Pre-trained embeddings
pretrained_emb = 'data/pkl/Emb.pkl'

def main(model_path, test_sets):

	nn = nlse.NLSE(pretrained_emb, model_file=model_path)

	for test_set in test_sets:
		print "\nTesting dataset: %s" % test_set
		with open(test_set, 'rb') as fid:
			X, Y = cPickle.load(fid) 			

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
				if not (j % 100):
					sys.stdout.write("\rTesting %d/%d" % (j+1, len(X)))
					sys.stdout.flush()   
			fm = Fmes.FmesSemEval(confusionMatrix=conf_mat)
			# set_trace()

			sys.stdout.write("\rAcc: %2.5f | Fm: %2.5f%%\n" % ( acc*100, fm*100))
			sys.stdout.flush()   

			
if __name__ == '__main__':
	MESSAGE = "python code/test.py model_path test_file_1 ... test_file_n"

	try:
		model_path = sys.argv[1]
		fnames = sys.argv[2:]   		
		if len(fnames) < 1:        	
			print "ERROR: No file names given\n"
			print MESSAGE         
		else:
			print "extracting features"            			
			main(model_path, fnames)			
	except IndexError:
		print "ERROR: missing arguments\n"
		print MESSAGE                     




