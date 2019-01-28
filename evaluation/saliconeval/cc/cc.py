#!/usr/bin/env python
#
# File Name : cc.py
#
# Description : Computes CC metric #

# Author : Ming Jiang

import numpy as np
import scipy.ndimage

def calc_score(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def compute_score(fixations, salMaps):
    """
    Computes AUC score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param res : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    score = []
    for fixation, salMap in zip(fixations, salMaps):
        # height, width = img.shape
        # mapheight, mapwidth = salMap.shape
        # salMap = scipy.ndimage.zoom(salMap, (float(height)/mapheight, float(width)/mapwidth), order=3)
        score.append(calc_score(fixation[0],salMap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)
