#!/usr/bin/env python
#
# File Name : nss.py
#
# Description : Computes NSS metric #

# Author : Ming Jiang
# Modified by: Bram van den Akker

import numpy as np
import scipy.ndimage

def calc_score(gtsAnn, resAnn):
    """
    Computer NSS score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : NSS score
    """

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.mean([ salMap[y-1][x-1] for y,x in gtsAnn ])

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