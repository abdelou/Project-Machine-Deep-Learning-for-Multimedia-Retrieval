import math
import cv2

import numpy as np

"""
This file contains distance functions that were given during practical sessions.
"""

def euclidean(l1,l2):
    distance = 0
    length = min(len(l1),len(l2))
    for i in range(length):
        distance += (l1[i] - l2[i])**2
    return math.sqrt(distance)
    
def chiSquareDistance(l1, l2):
    s = 0.0
    for i,j in zip(l1,l2):
        if i == j == 0.0:
            continue
        s += (i - j)**2 / (i + j)
    return s

def bhatta(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    num = np.sum(np.sqrt(np.multiply(l1,l2,dtype=np.float64)),dtype=np.float64)
    den = np.sqrt(np.sum(l1,dtype=np.float64)*np.sum(l2,dtype=np.float64))
    return math.sqrt( 1 - num / den )

def flann(a,b):
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)

def bruteForceMatching(a, b):
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 1e9
        
    # SIFT defaults to float32 and L2, ORB defaults to uint8 and Hamming
    # However, I quantized everything to uint8 to save space
    # So we use HAMMING if they are uint8 (ORB) or L2 if they are float.
    # Actually, SIFT quantized to uint8 still works better with L2 or L1.
    
    # Simple heuristic: if descriptor dim is 128, it's likely SIFT. if 32, it's ORB.
    norm = cv2.NORM_L2
    if a.shape[1] == 32: # ORB dim
        norm = cv2.NORM_HAMMING
        
    bf = cv2.BFMatcher(norm)
    try:
        matches = bf.match(a, b)
        if not matches: return 1e9
        dist_vals = [m.distance for m in matches]
        return np.mean(dist_vals)
    except:
        return 1e9


def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance = euclidean(l1, l2)
    elif distanceName in ["Correlation","Chicarre","Intersection","Bhattacharyya"]:
        if distanceName=="Correlation":
            methode=cv2.HISTCMP_CORREL
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Chicarre":
            distance = chiSquareDistance(np.float32(l1), np.float32(l2))
        elif distanceName=="Intersection":
            methode=cv2.HISTCMP_INTERSECT
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Bhattacharyya":
            distance = bhatta(np.float32(l1), np.float32(l2))
    elif distanceName=="Brute force":
        distance = bruteForceMatching(np.float32(l1), np.float32(l2))
    elif distanceName=="Flann":
        distance = flann(np.float32(l1), np.float32(l2))
    else:
        raise ValueError("Unknown distance")
    return distance