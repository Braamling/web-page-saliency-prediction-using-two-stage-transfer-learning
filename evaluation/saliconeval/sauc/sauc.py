#!/usr/bin/env python
#
# File Name : sauc.py
#
# Description : Computes shuffled AUC metric #

# Author : Ming Jiang

import numpy as np
import scipy.ndimage

def calc_score(gtsAnn, resAnn, shufMap, stepSize=.01):
    """
    Computer SAUC score. A simple implementation
    :param gtsAnn : list of fixation annotataions
    :param resAnn : list only contains one element: the result annotation - predicted saliency map
    :return score: int : score
    """


    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)

    Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ])
    Nfixations = len(gtsAnn)

    others = np.copy(shufMap)
    for y,x in gtsAnn:
        others[y-1][x-1] = 0

    ind = np.nonzero(others) # find fixation locations on other images
    nFix = shufMap[ind]
    randfix = salMap[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc


# def compute_score(fixations, salMaps):
#     """
#     Computes AUC score for a given set of predictions and fixations
#     :param gts : dict : fixation points with "image name" key and list of points as values
#     :param res : dict : salmap predictions with "image name" key and ndarray as values
#     :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
#     """
#     for fixation in fixations:
#         shufMap += fixation

#     score = []
#     for fixation, salMap in zip(fixations, salMaps):
#         score.append(calc_score(fixation[0],salMap, shufMap))
#     average_score = np.mean(np.array(score))
#     return average_score, np.array(score)
