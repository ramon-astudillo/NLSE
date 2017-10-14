def get_confusionMatrix(pred, gold):
    # Confusion Matrix
    # This assumes the order (neut-sent, pos-sent, neg-sent)
    mapp     = np.array([ 1, 2, 0])
    conf_mat = np.zeros((3, 3))
    for y, hat_y in zip(gold, pred):        
        conf_mat[mapp[y], mapp[hat_y]] += 1

    return conf_mat

def FmesSemEval(pred=None, gold=None, confusionMatrix=None):
    # This assumes the order (neut-sent, pos-sent, neg-sent)
    if confusionMatrix is None:
        assert pred is not None and gold is not None        
        confusionMatrix = get_confusionMatrix(pred, gold)
    
    # POS-SENT 
    # True positives pos-sent
    tp = confusionMatrix[1, 1]
    # False postives pos-sent
    fp = confusionMatrix[:, 1].sum() - tp
    # False engatives pos-sent
    fn = confusionMatrix[1, :].sum() - tp
    # Fmeasure binary
    FmesPosSent = Fmeasure(tp, fp, fn)

    # NEG-SENT 
    # True positives pos-sent
    tp = confusionMatrix[2, 2]
    # False postives pos-sent
    fp = confusionMatrix[:, 2].sum() - tp
    # False engatives pos-sent
    fn = confusionMatrix[2, :].sum() - tp
    # Fmeasure binary
    FmesNegSent = Fmeasure(tp, fp, fn)
 
    return (FmesPosSent + FmesNegSent)/2

def Fmeasure(tp, fp, fn):
    # Precision
    if tp+fp:
        precision = tp/(tp+fp)
    else:
        precision = 0 
    # Recall
    if tp+fn:
        recall    = tp/(tp+fn)
    else:
        recall    = 0
    # F-measure
    if precision + recall:
        return 2 * (precision * recall)/(precision + recall)
    else:
        return 0 

