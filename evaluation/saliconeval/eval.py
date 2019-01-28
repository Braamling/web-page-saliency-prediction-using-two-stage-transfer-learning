__author__ = 'shane-huang'
__version__ = '1.0'

import evaluation.saliconeval.nss.nss as nss
import evaluation.saliconeval.sauc.sauc as sauc
import evaluation.saliconeval.auc.auc as auc
import evaluation.saliconeval.cc.cc as cc
import numpy as np

def compute_scores(labels, predictions):
    score = {'nss': [], 'sauc': [], 'auc': [], 'cc': []}

    shufMap = np.zeros((64,64))
    for fixation in labels:
        shufMap += fixation[0]

    # score = {'nss': [], 'auc': [], 'cc': []}
    for label, prediction in zip(labels, predictions):
        fixations = np.argwhere(label[0] > 0.5)

        score['nss'].append(nss.calc_score(fixations, prediction))
        score['sauc'].append(sauc.calc_score(fixations, prediction, shufMap))
        score['auc'].append(auc.calc_score(fixations, prediction))
        score['cc'].append(cc.calc_score(label[0], prediction))
    avg_scores = {measure:np.mean(np.array(scores)) for (measure, scores) in score.items()}

    return avg_scores

