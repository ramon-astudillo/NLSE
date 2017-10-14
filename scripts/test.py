import cPickle
import nlse.FMeasure as Fmes
import numpy as np
import os
import sys
import argparse
import nlse.nlse as model


if __name__ == '__main__':

    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(prog='Tests a single model')
    parser.add_argument(
        '-o',
        help='Folder where the train data embeddings are',
        type=str,
        required=True
    )
    parser.add_argument(
        '-m',
        help='Path where model is saved', type=str,
        required=True
    )
    parser.add_argument(
        '-f',
        nargs='+',
        help='Test files', type=str,
        required=True
    )
    parser.add_argument(
        '-model',
        help='model used (MLP, GRU, CNN ...)',
        default="nlse",
        type=str
    )
    args = parser.parse_args(sys.argv[1:])

    # LOAD MODEL
    if args.model == 'nlse':
        nn = model.NN(None, None, model_file=args.m)
    else:
        raise NotImplementedError("Model %s not supported" % args.model)

    # Test sets
    test_sets = []
    for test_set in args.f:
        basename = os.path.basename(test_set).split('.txt')[0]
        test_sets.append('%s/%s.pkl' % (args.o, basename))

    # TEST ON EACH TEST SET
    results = []
    for test_set in test_sets:
        with open(test_set, 'rb') as fid:
            X, Y = cPickle.load(fid)
        acc = 0.
        mapp = np.array([1, 2, 0])
        conf_mat = np.zeros((3, 3))
        for j, x, y in zip(np.arange(len(X)), X, Y):
            # Prediction
            p_y = nn.forward(x)
            hat_y = np.argmax(p_y)
            # Confusion matrix
            conf_mat[mapp[y], mapp[hat_y]] += 1
            # Accuracy
            acc = (acc*j + (hat_y == y).astype(float))/(j+1)
            if not (j % 100):
                print "\rTesting %s %d/%d" % (test_set, j+1, len(X)),
                sys.stdout.flush()
        fm = Fmes.FmesSemEval(confusionMatrix=conf_mat)
        results.append((acc, fm))
        print ""

    # Display result and store along with model
    results_path = '%s/%s.results' % (
        os.path.dirname(args.m),
        os.path.basename(args.m).split('.pkl')[0]
    )
    with open(results_path, 'w') as fid:
        print "\nTest-Set\tAcc\tF-meas"
        fid.write("%15s\t%10s\t%10s\n" % ("Test-Set", "Accuracy", "F-measure"))
        for test_set, result in zip(args.f, results):
            basename = os.path.basename(test_set)
            acc, fm = result
            print "%15s\t%10.2f%%\t%10.3f" % (basename, acc*100, fm)
            fid.write("%s\t%2.2f%%\t%1.3f\n" % (basename, acc*100, fm))
    print "\nResults are stored under %s" % results_path
