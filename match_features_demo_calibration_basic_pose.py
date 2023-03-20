import argparse
from pathlib import Path

import cv2
import numpy as np
#import tensorflow as tf  # noqa: E402
## torch
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from settings import EXPER_PATH  # noqa: E402

from utils.loader import dataLoader, modelLoader, pretrainedLoader
import torch.optim as optim
from utils.utils import getPtsFromHeatmap, flattenDetection
#from utils.utils import box_nms
from utils.d2s import DepthToSpace, SpaceToDepth

import os


import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def extract_SIFT_keypoints_and_descriptors(img, m=None, mask=None, feature='sift', green=False
                                           , o=3, c=0.04, e=10, s=1.6
                                           , sf=1.2, nl=8):
    if not green:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img[:, :, 1]
    if 'orb' in feature or 'akaze' in feature:
        det = cv2.ORB_create(nfeatures=10000, scaleFactor=sf, nlevels=nl)
    else:
        det = cv2.xfeatures2d.SIFT_create(nfeatures=10000, nOctaveLayers=o, contrastThreshold=c, edgeThreshold=e, sigma=s)
    kp, desc = det.detectAndCompute(np.squeeze(gray_img), mask=mask)

    return kp, desc


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=10000, m=None, mask=None):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    if m!=None:
        keyp=np.array([])
        for k in keypoints:
            if mask[int(k[0]),int(k[1])]>127:
                if len(keyp)==0:
                    keyp=np.array([k])
                else:
                    keyp=np.append(keyp,np.array([k]),axis=0)
        keypoints=keyp
    if len(keypoints)>0:
        keypoints = select_k_best(keypoints, keep_k_points)
        keypoints = keypoints.astype(int)

        # Get descriptors for keypoints
        desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]
    else:
        desc=np.array([])

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc

def extract_superpoint_kp_and_desc_pytorch(
        keypoint_map, descriptor_map, conf_thresh=0.015,
        nms_dist=4, cell=8, device='cpu',
        keep_k_points=10000, m=None, mask=None):

    def select_k_best(points, k, desc):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        pointsdesc=np.append(points,desc,axis=1)
        sorted_prob = pointsdesc[pointsdesc[:, 2].argsort(), :]#:2
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :2], sorted_prob[-start:, 3:]

    semi_flat_tensor = flattenDetection(keypoint_map[0, :, :, :]).detach()
    semi_flat = toNumpy(semi_flat_tensor)
    semi_thd = np.squeeze(semi_flat, 0)
    #print(semi_thd.shape,np.amax(semi_thd))
    pts=getPtsFromHeatmap(
            semi_thd, conf_thresh=conf_thresh, nms_dist=nms_dist)
    #cv2.imwrite("/home/leon/Experiments/output/endomapper/MICCAI2022/examples_paper/semi_thd_"+str(pts.shape[1])+".png",semi_thd/np.amax(semi_thd)*255.)
    print(pts.shape)
    #exit(1)
    desc_sparse_batch = sample_desc_from_points(
            descriptor_map, pts, cell=cell, device=device)
    #print(pts.shape, desc_sparse_batch.shape)

    # Extract keypoints
    keypoints = np.transpose(pts)[:,[1,0,2]]
    descriptors = np.transpose(desc_sparse_batch)
    if m!=None:
        keyp=np.array([])
        desc=np.array([])
        for i in range(keypoints.shape[0]):
            if mask[int(keypoints[i,0]),int(keypoints[i,1])]>127:
                if len(keyp)==0:
                    keyp=np.array([keypoints[i,:]])
                    desc=np.array([descriptors[i,:]])
                else:
                    keyp=np.append(keyp,np.array([keypoints[i,:]]),axis=0)
                    desc=np.append(desc,np.array([descriptors[i,:]]),axis=0)
        keypoints=keyp
        descriptors=desc
    if len(keypoints)>0:
        keypoints, descriptors = select_k_best(keypoints, keep_k_points, descriptors)
        keypoints = keypoints.astype(int)
    else:
        descriptors=np.array([])

    # Convert from just pts to cv2.KeyPoints
    keypoints_cv2 = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints_cv2, descriptors


def match_descriptors_cv2_BF(kp1, desc1, kp2, desc2, feature='sift'):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if 'orb' in feature or 'akaze' in feature:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches2 = bf.match(desc1, desc2)
    #l=[]
    matches=[]
    for m in matches2:
        d1=desc1[m.queryIdx,:]
        d2=desc2[m.trainIdx,:]
        distance = d1 @ d2
        if distance>0.8:
            matches.append(m)
        #l.append(distance)
    #plt.hist(l,bins=100)#,cumulative=True
    #plt.savefig('/home/leon/Experiments/output/hist.png')
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches
    

def match_descriptors_cv2_ratio(kp1, desc1, kp2, desc2, feature='sift', ratio=0.75):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if 'orb' in feature or 'akaze' in feature:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    #matches = bf.match(desc1, desc2)
    matches = bf.knnMatch(desc1,desc2,k=2)
    # Apply ratio test
    good = []
    #print(len(matches),len(matches[0]))
    if len(matches[0])==1:
        for m in matches:
            good.append(m[0])
    else:
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append(m)
    matches=good
    #print(len(matches),len(matches[0]))
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]
    
    return m_kp1, m_kp2, matches
    
    
def nn_match_two_way(kp1, des1, kp2, des2, feature='sift', nn_thresh=1.5):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.
    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    
    desc1=des1.T
    desc2=des2.T
    
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    
    kp1_m=np.array(kp1)[m_idx1]
    kp2_m=np.array(kp2)[m_idx2]
    matches_cv2=[]
    for i in range(len(m_idx1)):
        matches_cv2.append(cv2.DMatch(m_idx1[i],m_idx2[i],0,scores[i]))
    
    return kp1_m.tolist(), kp2_m.tolist(), matches_cv2
    
def estimateRotAndTranslFromEssentialMatrix(E_21):
    """
    Descompose the essential matrix between two view to estimate the relative traslation
    and rotation between them. It provides 2 possible rotations, it is necessary to validate
    which one is correct.

    -input:
        E_21: 3x3 matrix -> Essential Matrix
    -output:
        t_21: 3 vector -> relative traslation vector
        RA_21: 3x3 matrix -> relative rotation matrix
        RB_21: 3x3 matrix -> relative rotation matrix
    """
    W = np.array([[0,-1,0],
                  [1,0,0],
                  [0,0,1]])
    print(E_21)
    u, s, vh = np.linalg.svd(E_21)
    RA_21 = u @ W @ vh
    RB_21 = u @ W.T @ vh
    t_21 = u @ np.array([[0],[0],[1]])
    return t_21, RA_21, RB_21

def distanceCheckComp(lines,pts1,pts2,thresholdDist=20):
    c=1440
    good = None
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        p1=np.array([x0,y0])
        p2=np.array([x1,y1])
        d=np.abs(np.cross(p2-p1, p1-pt1)) / np.linalg.norm(p2-p1)
        if good is None:
            good = np.array([[int(d<=thresholdDist)]],dtype='uint8')
        else:
            good = np.append(good,np.array([[int(d<=thresholdDist)]],dtype='uint8'), axis=0)
    return good

def compute_inliers_3(matched_kp1, matched_kp2, geom_model='H', video='0', Rot=None, Trans=None):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    if 'calibrated05' in video:
        dist = np.array([-0.28650, 0.29524, -0.00212, 0.00152]).reshape((4, 1))
        mtx = np.array([[530.90002, 0.0, 136.63037],
                        [0.0, 581.00362, 161.32884],
                        [0.0, 0.0, 1.0]])
    elif 'calibrated11' in video:
        dist = np.array([-0.196312, 0.129540, 0.004356, 0.006236]).reshape((4, 1))
        mtx = np.array([[391.656525, 0.0, 165.964371],
                        [0.0, 426.835144, 154.498138],
                        [0.0, 0.0, 1.0]])
    elif 'calibrated16' in video or 'calibrated17' in video:
        dist = np.array([-0.186853, 0.122769, -0.010146, -0.003869]).reshape((4, 1))
        mtx = np.array([[755.312744, 0.0, 327.875000],
                        [0.0, 420.477722, 165.484406],
                        [0.0, 0.0, 1.0]])
    else:
        dist = np.array([-0.1205372, -0.01179983, 0.00269742, -0.0001362505]).reshape((4, 1)) 
        ##calibration from HCULB_00044,(45 and 48 are good choices too)
        mtx = np.array([[733.1061, 0.0, 739.2826],
                        [0.0, 735.719, 539.6911],
                        [0.0, 0.0, 1.0]])
    und1 = np.float32(np.zeros(matched_pts1.shape)[:, np.newaxis, :])
    cv2.undistortPoints(src=matched_pts1[:, np.newaxis, :], cameraMatrix=mtx, distCoeffs=dist, dst=und1, P=mtx)
    und2 = np.float32(np.zeros(matched_pts2.shape)[:, np.newaxis, :])
    cv2.undistortPoints(src=matched_pts2[:, np.newaxis, :], cameraMatrix=mtx, distCoeffs=dist, dst=und2, P=mtx)

    # Estimate the homography between the matches using RANSAC
    H = np.eye(3)
    inliers_H = np.array([0])
    '''cv2.findHomography(und1[:,0, [1, 0]],
                                    und2[:,0, [1, 0]],
                                    cv2.RANSAC,
                                    ransacReprojThreshold = 3.,
                                    confidence = 0.9999)'''
    if inliers_H is None:
        inliers_H = np.array([0])
    # Estimate the essential matrix between the matches using RANSAC
    Es, inliers_E = cv2.findEssentialMat(und1[:, 0, [1, 0]],
                                         und2[:, 0, [1, 0]],
                                         mtx,
                                         cv2.RANSAC,
                                         prob=0.9999,
                                         threshold=1.)
    if inliers_E is None:
        inliers_E = np.array([0])
        # inliers_Ef = np.array([0])
        inliers_Ec = np.array([0])
        R = np.eye(3)
        R2 = np.eye(3)
        t = np.eye(3, 1)
    else:
        E = Es[:3, :]
        n = 1
        while np.isnan(np.sum(E)):
            E = Es[n * 3:(n + 1) * 3]
            n += 1
        inliers_Ef = inliers_E.copy()
        _, _, _, _ = cv2.recoverPose(E, und1[:, 0, [1, 0]], und2[:, 0, [1, 0]], mtx, mask=inliers_Ef)
        R, R2, t = cv2.decomposeEssentialMat(E)  # estimateRotAndTranslFromEssentialMatrix(Ess)#

    Trans_cross = np.array([[0., -Trans[2, 0], Trans[1, 0]],
                            [Trans[2, 0], 0., -Trans[0, 0]],
                            [-Trans[1, 0], Trans[0, 0], 0.]])
    Ec = Trans_cross @ Rot

    # We select only inlier points
    pts1 = und1[:, 0, :]  # cv2.KeyPoint_convert(undi1)#
    pts2 = und2[:, 0, :]  # cv2.KeyPoint_convert(undi2)#

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    Fc = np.linalg.inv(mtx.T) @ Ec @ np.linalg.inv(mtx)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, Fc)
    lines2 = lines2.reshape(-1, 3)
    # print(lines2.shape, pts2.shape,pts1.shape)
    # inliers_Ec = distanceCheckComp(lines2,pts2,pts1,3)
    inliers_Ef = distanceCheckComp(lines2, pts2, pts1, 5)
    # print(np.sum(inliers_Ef),np.sum(inliers_Ec))
    inliers_Ec = inliers_E.copy()
    _, _, _, _, _ = cv2.recoverPose(Ec, und1[:, 0, [1, 0]], und2[:, 0, [1, 0]], mtx, distanceThresh=1000.,
                                    mask=inliers_Ec)

    inliers_H = inliers_H.flatten()
    inliers_E = inliers_E.flatten()
    inliers_Ef = inliers_Ef.flatten()
    inliers_Ec = inliers_Ec.flatten()
    return inliers_H, inliers_E, inliers_Ef, inliers_Ec, R, R2, t, Fc

def compute_inliers_2(matched_kp1, matched_kp2, geom_model='H',video='0',Rot=None,Trans=None):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    if 'calibrated05' in video:
        dist = np.array([ -0.28650, 0.29524, -0.00212, 0.00152]).reshape((4,1)) 
        mtx =  np.array([[530.90002, 0.0, 136.63037],
                 [0.0, 581.00362, 161.32884],
                 [0.0, 0.0, 1.0]])
    elif 'calibrated11' in video:
        dist = np.array([ -0.196312, 0.129540, 0.004356, 0.006236]).reshape((4,1)) 
        mtx =  np.array([[391.656525, 0.0, 165.964371],
                 [0.0, 426.835144, 154.498138],
                 [0.0, 0.0, 1.0]])
    elif 'calibrated16' in video or 'calibrated17' in video:
        dist = np.array([ -0.186853, 0.122769, -0.010146, -0.003869]).reshape((4,1)) 
        mtx =  np.array([[755.312744, 0.0, 327.875000],
                 [0.0, 420.477722, 165.484406],
                 [0.0, 0.0, 1.0]])
    else:
        dist = np.array([ -0.1205372, -0.01179983, 0.00269742, -0.0001362505]).reshape((4,1)) ##calibration from HCULB_00044,(45 and 48 are good choices too)
        mtx =  np.array([[733.1061, 0.0, 739.2826],
                 [0.0, 735.719, 539.6911],
                 [0.0, 0.0, 1.0]])
    und1 = np.float32(np.zeros(matched_pts1.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts1[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und1,P=mtx)
    und2 = np.float32(np.zeros(matched_pts2.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts2[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und2,P=mtx)
    
    # Estimate the homography between the matches using RANSAC
    H, inliers_H = cv2.findHomography(und1[:,0, [1, 0]],
                                    und2[:,0, [1, 0]],
                                    cv2.RANSAC,
                                    ransacReprojThreshold = 3.,
                                    confidence = 0.9999)
    if inliers_H is None:
        inliers_H = np.array([0])
    # Estimate the essential matrix between the matches using RANSAC
    Es, inliers_E = cv2.findEssentialMat(und1[:,0, [1, 0]],
                                    und2[:,0, [1, 0]],
                                    mtx,
                                    cv2.RANSAC,
                                    prob = 0.9999,
                                    threshold = 3.)
    if inliers_E is None:
        inliers_E = np.array([0])
        inliers_Ef = np.array([0])
        inliers_Ec = np.array([0])
        R = np.eye(3)
        R2 = np.eye(3)
        t = np.eye(3,1)
        Fc = np.eye(3)
    else:
        E = Es[:3,:]
        n = 1
        while np.isnan(np.sum(E)):
            E = Es[n*3:(n+1)*3]
            n += 1
        inliers_Ef = inliers_E.copy()
        _, _, _, _ = cv2.recoverPose(E, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, mask=inliers_Ef)
        R, R2, t = cv2.decomposeEssentialMat(E)#estimateRotAndTranslFromEssentialMatrix(Ess)#
    
                                        
        Trans_cross = np.array([[0.,-Trans[2,0],Trans[1,0]],
                    [Trans[2,0],0.,-Trans[0,0]],
                    [-Trans[1,0],Trans[0,0],0.]])
        Ec = Trans_cross @ Rot
        inliers_Ec = inliers_E.copy()
        print(inliers_Ec.shape, np.sum(inliers_Ec))
        _, _, _, _, _ = cv2.recoverPose(Ec, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, distanceThresh=1000., mask=inliers_Ec)
        print(inliers_Ec.shape, np.sum(inliers_Ec))
        Fc = np.linalg.inv(mtx.T) @ Ec @ np.linalg.inv(mtx)
    
    inliers_H = inliers_H.flatten()
    inliers_E = inliers_E.flatten()
    inliers_Ef = inliers_Ef.flatten()
    inliers_Ec = inliers_Ec.flatten()
    return inliers_H, inliers_E, inliers_Ef, inliers_Ec, R, R2, t, Fc


def compute_inliers(matched_kp1, matched_kp2, geom_model='H',video='0',Rot=None,Trans=None):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    if 'calibrated05' in video:
        dist = np.array([ -0.28650, 0.29524, -0.00212, 0.00152]).reshape((4,1)) 
        mtx =  np.array([[530.90002, 0.0, 136.63037],
                 [0.0, 581.00362, 161.32884],
                 [0.0, 0.0, 1.0]])
    elif 'calibrated11' in video:
        dist = np.array([ -0.196312, 0.129540, 0.004356, 0.006236]).reshape((4,1)) 
        mtx =  np.array([[391.656525, 0.0, 165.964371],
                 [0.0, 426.835144, 154.498138],
                 [0.0, 0.0, 1.0]])
    elif 'calibrated16' in video or 'calibrated17' in video:
        dist = np.array([ -0.186853, 0.122769, -0.010146, -0.003869]).reshape((4,1)) 
        mtx =  np.array([[755.312744, 0.0, 327.875000],
                 [0.0, 420.477722, 165.484406],
                 [0.0, 0.0, 1.0]])
    else:
        dist = np.array([ -0.1205372, -0.01179983, 0.00269742, -0.0001362505]).reshape((4,1)) ##calibration from HCULB_00044,(45 and 48 are good choices too)
        mtx =  np.array([[733.1061, 0.0, 739.2826],
                 [0.0, 735.719, 539.6911],
                 [0.0, 0.0, 1.0]])
    und1 = np.float32(np.zeros(matched_pts1.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts1[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und1,P=mtx)
    und2 = np.float32(np.zeros(matched_pts2.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts2[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und2,P=mtx)
    if 'E' in geom_model:
        # Estimate the essential matrix between the matches using RANSAC
        T, inliers = cv2.findEssentialMat(und1[:,0, [1, 0]],
                                        und2[:,0, [1, 0]],
                                        mtx,
                                        cv2.RANSAC,
                                        prob = 0.9999,
                                        threshold = 3.)
    '''elif 'F' in geom_model:
        # Estimate the fundamental matrix between the matches using RANSAC
        T, inliers = cv2.findFundamentalMat(und1[:,0, [1, 0]],
                                        und2[:,0, [1, 0]],
                                        cv2.RANSAC,
                                        ransacReprojThreshold = 3.,
                                        confidence = 0.9999)
    else:
        # Estimate the homography between the matches using RANSAC
        T, inliers = cv2.findHomography(und1[:,0, [1, 0]],
                                        und2[:,0, [1, 0]],
                                        cv2.RANSAC,
                                        ransacReprojThreshold = 3.,
                                        confidence = 0.9999)
    if inliers is None:
        inliers=[0]
        R = np.eye(3)
        t = np.array([0,0,0]).T
    else:
        if T.shape[0]>3:
            T=T[:3,:]
        _, R, t, _,_ = cv2.recoverPose(T, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, distanceThresh=3, mask=inliers)
        inliers = inliers.flatten()
    return T, inliers, R, t'''
    '''inliers = inliers.astype(bool).flatten()

    # OpenCV can return multiple values as 6x3 or 9x3 matrices
    if T is None:
        return _fail()
    elif T.shape[0] != 3:
        Es = np.split(T, len(T) / 3)
    # Or a single matrix
    else:
        Es = [T]

    # Find the best E
    E, num_inlier, R, t = None, 0, np.eye(3), np.zeros((3,1))
    # mask_E_cheirality_check = None
    for _E in Es:
        _num_inlier, _R, _t, _mask = cv2.recoverPose(_E, und1[:,0, [1, 0]][inliers],
                                                     und2[:,0, [1, 0]][inliers])
        if _num_inlier >= num_inlier:
            num_inlier = _num_inlier
            E = _E
            R = _R
            t = _t
            # This is unused for now
            # mask_E_cheirality_check = _mask.flatten().astype(bool)

    indices = inliers.flatten()
    return E, indices, R ,t'''
    
                                        
    Trans_cross = np.array([[0.,-Trans[2,0],Trans[1,0]],
                [Trans[2,0],0.,-Trans[0,0]],
                [-Trans[1,0],Trans[0,0],0.]])
    Ess = Trans_cross @ Rot
    
    #inliers=np.ones((und1[:,0, [1, 0]].shape[0],1),dtype=np.uint8)
    _, R, t, _= cv2.recoverPose(Ess, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, mask=inliers.copy())
    R, R2, t = cv2.decomposeEssentialMat(Ess)#estimateRotAndTranslFromEssentialMatrix(Ess)#
    inliers = inliers.flatten()
    
    return Ess, inliers, R, R2, t


def preprocess_image(img_file, img_size, green=False):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    #img = cv2.resize(img, img_size)
    img_orig = img#.copy()

    if not green:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = img[:,:,1]
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed[:,205:-155], img_orig

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

def sample_desc_from_points(coarse_desc, pts, cell=8, device='cpu'):
    # --- Process descriptor.
    H, W = coarse_desc.shape[2]*cell, coarse_desc.shape[3]*cell
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.to(device)
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc
    

def inverse_mapping(xmap,ymap):
    #h, w, _ = rectilinear.shape
    #H, W, _ = equirectangular_image.shape
    h = 1080
    w = 1440
    H = 1080
    W = 1440
    map1_inverse = np.zeros((H, W))
    map2_inverse = np.zeros((H, W))

    s = time.time()

    data = []
    coords = []
    for j in range(w):
        for i in range(h):
            data.append([xmap[i, j], ymap[i, j]])
            coords.append((i, j))
    data = np.array(data)
    tree = cKDTree(data, leafsize=16, compact_nodes=True, balanced_tree=True)
    coords.append((0, 0))  # extra coords for failed neighbour search

    e1 = time.time()
    print("Tree creation took {:0.2f} seconds".format(e1-s))

    x = np.linspace(0.0, W, num=W, endpoint=False)
    y = np.linspace(0.0, H, num=H, endpoint=False)
    pts = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    distances, indices = tree.query(pts, k=5, p=2, distance_upper_bound=5.0)

    e2 = time.time()
    print("Tree query took {:0.2f} seconds".format(e2-e1))

    # TODO optimization (any suggestions? :S)
    for (x, y), ds, idxs in zip(pts.astype(np.uint16), distances, indices):
        wsum_i = 0
        wsum_j = 0
        wsum = np.finfo(float).eps
        for d, idx in zip(ds, idxs):
            if d==0:
                continue
            w = 1.0 / (d*d)
            wsum += w
            wsum_i += w*coords[idx][0]
            wsum_j += w*coords[idx][1]
        wsum_i /= wsum
        wsum_j /= wsum
        map1_inverse[y, x] = wsum_j
        map2_inverse[y, x] = wsum_i

    e3 = time.time()
    print("Weighted sums took {:0.2f} seconds".format(e3-e2))
    
    return map1_inverse, map2_inverse

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t


def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q
    
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[0:2]
    #img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    #img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        #img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
    
def distanceCheck(img1,lines,pts1,pts2,thresholdDist=20):
    r,c = img1.shape[0:2]
    good = np.array([],dtype=int)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        p1=np.array([x0,y0])
        p2=np.array([x1,y1])
        d=np.abs(np.cross(p2-p1, p1-pt1)) / np.linalg.norm(p2-p1)
        good = np.append(good,np.array([int(d<=thresholdDist)]))
    return good
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('weights_name', type=str)
    parser.add_argument('img1_path', type=str)
    parser.add_argument('img2_path', type=str)
    parser.add_argument('--m', type=str, default=None,
                        help='Mask to ignore matches.')
    parser.add_argument('--H', type=int, default=480,
                        help='The height in pixels to resize the images to. \
                                (default: 480)')
    parser.add_argument('--W', type=int, default=640,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=10000,
                        help='Maximum number of keypoints to keep \
                        (default: 10000)')
    parser.add_argument('--conf_thresh', type=int, default=0.015,
                        help='Detection confidence threshold \
                        (default: 0.015)')
    parser.add_argument('--ratio', type=int, default=80,
                        help='Ratio for matching out of 100 \
                        (default: 80)')
    parser.add_argument('--o', type=int, default=3,
                        help='SIFT parameter nOctaveLayers \
                        (default: 3)')
    parser.add_argument('--c', type=int, default=40,
                        help='SIFT parameter contrastThreshold \
                        (default: 10)')
    parser.add_argument('--e', type=int, default=10,
                        help='SIFT parameter edgeThreshold \
                        (default: 10)')
    parser.add_argument('--s', type=int, default=16,
                        help='SIFT parameter sigma \
                        (default: 16)')
    parser.add_argument('--sf', type=int, default=12,
                        help='ORB parameter scaleFactor \
                        (default: 12)')
    parser.add_argument('--nl', type=int, default=8,
                        help='ORB parameter nlevels \
                        (default: 8)')
    parser.add_argument('--nms', type=int, default=1,
                        help='Non-maximum supression size \
                        (default: 1)')
    parser.add_argument('--green', action='store_true', default=False,
                         help='Use only the green channel of the images.')
    parser.add_argument('--spec', action='store_true', default=False,
                         help='Filter out features on top of specularities.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path.')
    args = parser.parse_args()

    weights_name = args.weights_name
    img1_file = args.img1_path
    img2_file = args.img2_path
    #print(img1_file)
    #print(img2_file)
    if args.m!=None:
        mask=cv2.imread(args.m,cv2.IMREAD_GRAYSCALE)
    else:
        mask=None
    img_size = (args.W, args.H)
    keep_k_best = args.k_best
    conf=args.conf_thresh
    nms=args.nms
    green=args.green
    spec=args.spec
    output=args.output
    ratio = args.ratio / 100.

    weights_root_dir = Path(EXPER_PATH)
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    if 'superpoint' in weights_name or 'sp_v6' in weights_name:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#"cpu")#
        if weights_dir.suffix!='.pth':
            net = modelLoader(model='SuperPointNet_notrack')
            if torch.cuda.is_available():
                net.cuda()
            optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999))
            net, _, _ = pretrainedLoader(
                    net=net, optimizer=optimizer, epoch=1, path=weights_dir, full_path=True
            )
        else:
            net = modelLoader(model='SuperPointNet_pretrained')
            if torch.cuda.is_available():
                net.cuda()
            optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999))
            net, _, _ = pretrainedLoader(
                    net=net, optimizer=optimizer, epoch=1, path=weights_dir, mode='other', full_path=True
            )
        net.eval()
        torch.no_grad()

    img1, img1_orig = preprocess_image(img1_file, img_size, green)
    img1_color=img1_orig[:,205:-155]
    #cv2.imwrite("/home/leon/Experiments/output/endomapper/MIDL2022/img_from_basic.png",img1*255.)
    img2, img2_orig = preprocess_image(img2_file, img_size, green)
    img2_color=img2_orig[:,205:-155]
        
    if spec:
        img1_gray=cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        ret, img1_mask=cv2.threshold(img1_gray, 180, 255, cv2.THRESH_BINARY)
        kp1spec=np.array([],dtype='int')
        img2_gray=cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
        ret, img2_mask=cv2.threshold(img2_gray, 180, 255, cv2.THRESH_BINARY)
        kp2spec=np.array([],dtype='int')
    if 'superpoint' in weights_name or 'sp_v6' in weights_name:
        out1=net(torch.from_numpy(np.expand_dims(img1, 0)).permute(0,3,1,2).to(device))
        #cv2.imwrite("/home/leon/Experiments/output/endomapper/MIDL2022/semi_from_basic.png",out1['semi'].detach().cpu().numpy()[0,:-1,:,:].reshape(1,135*8,135*8)[0,:,:]*255.)
        if weights_dir.suffix!='.pth':
            #print(out1['semi'].shape,out1['desc'].shape,device)
            kp1, desc1 = extract_superpoint_kp_and_desc_pytorch(
                    out1['semi'], out1['desc'], conf_thresh=conf,
                    nms_dist=nms, cell=8, device=device,
                    keep_k_points=keep_k_best, m=args.m, mask=mask)
        else:
            #print(out1[0].shape,out1[1].shape,device)
            kp1, desc1 = extract_superpoint_kp_and_desc_pytorch(
                    out1[0], out1[1], conf_thresh=conf,
                    nms_dist=nms, cell=8, device=device,
                    keep_k_points=keep_k_best, m=args.m, mask=mask)
        out2=net(torch.from_numpy(np.expand_dims(img2, 0)).permute(0,3,1,2).to(device))
        if weights_dir.suffix!='.pth':
            #print(out2['semi'].shape,out2['desc'].shape,device)
            kp2, desc2 = extract_superpoint_kp_and_desc_pytorch(
                    out2['semi'], out2['desc'], conf_thresh=conf,
                    nms_dist=nms, cell=8, device=device,
                    keep_k_points=keep_k_best, m=args.m, mask=mask)
        else:
            #print(out2[0].shape,out2[1].shape,device)
            kp2, desc2 = extract_superpoint_kp_and_desc_pytorch(
                    out2[0], out2[1], conf_thresh=conf,
                    nms_dist=nms, cell=8, device=device,
                    keep_k_points=keep_k_best, m=args.m, mask=mask)
        #print((torch.from_numpy(np.expand_dims(img1, 0)).permute(0,3,1,2) - torch.from_numpy(np.expand_dims(img2, 0)).permute(0,3,1,2)).abs().max(),(out1['conv'] - out2['conv']).abs().max(),(out1['bn'] - out2['bn']).abs().max(),(out1['relu'] - out2['relu']).abs().max())
    else:
        kp1, desc1 = extract_SIFT_keypoints_and_descriptors(img1_color, args.m, mask, weights_name, green, o=args.o,c=args.c/1000,e=args.e,s=args.s/10,sf=args.sf/10,nl=args.nl)
        kp2, desc2 = extract_SIFT_keypoints_and_descriptors(img2_color, args.m, mask, weights_name, green, o=args.o,c=args.c/1000,e=args.e,s=args.s/10,sf=args.sf/10,nl=args.nl)
    
    if kp1 is None:
        kp1=np.array([])
        desc1=np.array([])
    if spec and len(kp1)>0:
        for s in range(len(kp1)):
            if img1_mask[int(kp1[s].pt[1]),int(kp1[s].pt[0])]<127:
                kp1spec=np.append(kp1spec,[s])
        kp1=[kp1[s] for s in kp1spec]
        desc1=desc1[kp1spec,:]
    if kp2 is None:
        kp2=np.array([])
        desc2=np.array([])
    if spec and len(kp2)>0:
        for s in range(len(kp2)):
            if img2_mask[int(kp2[s].pt[1]),int(kp2[s].pt[0])]<127:
                kp2spec=np.append(kp2spec,[s])
        kp2=[kp2[s] for s in kp2spec]
        desc2=desc2[kp2spec,:]
    kp1u=[]
    kp1scaled=[]
    for kp in kp1:
        kp1u.append(cv2.KeyPoint(kp.pt[0],kp.pt[1],kp.size))
        kp1scaled.append(cv2.KeyPoint(kp.pt[0]*256./1080.,kp.pt[1]*256./1080.,2))#kp.size
        x=kp.pt[0]
        y=kp.pt[1]
        #x=x*288/256+216
        #y=y*288/256
        #x=x*1080/256+205
        #y=y*1080/256
        x=x+205
        y=y
        kp.pt=(int(x),int(y))
    kp2u=[]
    kp2scaled=[]
    for kp in kp2:
        kp2u.append(cv2.KeyPoint(kp.pt[0],kp.pt[1],kp.size))
        kp2scaled.append(cv2.KeyPoint(kp.pt[0]*256./1080.,kp.pt[1]*256./1080.,2))#kp.size
        x=kp.pt[0]
        y=kp.pt[1]
        #x=x*288/256+216
        #y=y*288/256
        #x=x*1080/256+205
        #y=y*1080/256
        x=x+205
        y=y
        kp.pt=(int(x),int(y))
        
    img1_name = os.path.basename(os.path.normpath(img1_file))
    img2_name = os.path.basename(os.path.normpath(img2_file))
    Ts = np.load("/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/00034/25/RandT.npy")
    RandT = open("/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/00034/25/RandT.txt", "r")
    lines = RandT.read().splitlines()
    numbers = [int(x[33:-4]) for x in lines]
    zipped=zip(numbers, Ts[:])
    sorted_zip=sorted(zipped)
    tuples=zip(*sorted_zip)
    images_number, T = [list(tuple) for tuple in tuples]

    #hello my friendddddddd
    image_number1 = int(img1_file[-7:-4])
    image_number2 = int(img2_file[-7:-4])
    # Find out which is the closest ground truth
    if image_number1 >= images_number[-1]:
        index_T1 = len(images_number)-1
    else:
        index_previous = len(images_number)-1
        while images_number[index_previous] > image_number1 and index_previous > 0:
            index_previous -= 1
        if abs(image_number1-images_number[index_previous]) <= abs(image_number1-images_number[index_previous+1]):
            index_T1 = index_previous
        else:
            index_T1 = index_previous+1
            
    if image_number2 >= images_number[-1]:
        index_T2 = len(images_number)-1
    else:
        index_previous = len(images_number)-1
        while images_number[index_previous] > image_number2 and index_previous > 0:
            index_previous -= 1
        if abs(image_number2-images_number[index_previous]) <= abs(image_number2-images_number[index_previous+1]):
            index_T2 = index_previous
        else:
            index_T2 = index_previous+1
    maxim=max(abs(image_number1-images_number[index_T1]),abs(image_number2-images_number[index_T2]))
    if maxim>5:
        print(abs(image_number1-images_number[index_T1]),abs(image_number2-images_number[index_T2]))
    
    # Get R, t from calibration information
    R_1, t_1 = T[index_T1][0:3,0:3], T[index_T1][0:3,3].reshape((3, 1))
    R_2, t_2 = T[index_T2][0:3,0:3], T[index_T2][0:3,3].reshape((3, 1))
    
    # Compute dR, dt
    dR = np.dot(R_2, R_1.T)
    dT = t_2 - np.dot(dR, t_1)
    #print(np.linalg.norm(R_1),np.linalg.norm(R_2),np.linalg.norm(dR),np.linalg.norm(R_E))
    #print(np.linalg.norm(t_1),np.linalg.norm(t_2),np.linalg.norm(dT),np.linalg.norm(t_E))
    
    # Match and get rid of outliers
    if len(kp1)>0 and len(kp2)>0:
        if 'superpoint' in weights_name or 'sp_v6' in weights_name:
            m_kp1, m_kp2, matches = match_descriptors_cv2_BF(kp1, desc1.astype('float32'), kp2, desc2.astype('float32'), weights_name)#nn_match_two_way
        elif 'sift' in weights_name:
            m_kp1, m_kp2, matches = match_descriptors_cv2_ratio(kp1, desc1, kp2, desc2, weights_name, ratio)
        else:
            m_kp1, m_kp2, matches = match_descriptors_cv2_BF(kp1, desc1, kp2, desc2, weights_name)#nn_match_two_way
        if len(matches)>0:
            #H, inliers_H, R_H, t_H = compute_inliers(m_kp1, m_kp2, 'H', img1_file)
            #F, inliers_F, R_F, t_F = compute_inliers(m_kp1, m_kp2, 'F', img1_file)
            inliers_H, inliers_E, inliers_Ef, inliers_Ec, R_E, R_E2, t_E, Fc = compute_inliers_3(m_kp1, m_kp2, 'E', img1_file,dR,dT)
        else:
            #inliers_H=[0]
            #inliers_F=[0]
            inliers_E=[0]
    else:
        matches=[]
        #inliers_H=[0]
        #inliers_F=[0]
        inliers_E=[0]
    #if np.amax(inliers_H)>1:
        #inliers_H=[0]
    #if np.amax(inliers_F)>1:
        #inliers_F=[0]
    if np.amax(inliers_E)>1:
        inliers_E=[0]
        
    '''dist = np.array([ -0.1205372, -0.01179983, 0.00269742, -0.0001362505]).reshape((4,1)) ##calibration from HCULB_00044 (45 and 48 are good choices too)
    mtx =  np.array([[733.1061, 0.0, 739.2826],
             [0.0, 735.719, 539.6911],
             [0.0, 0.0, 1.0]])
    #cv2.imwrite("/home/leon/Experiments/output/endomapper/SEP2021/undist.png",img2_orig)
    h, w, c = img2_orig.shape
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (w, h), np.eye(3), balance=1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), new_K, (w, h), cv2.CV_16SC2)#new_K instead of the second mtx
    xmap, ymap=cv2.convertMaps(map1,map2,cv2.CV_32FC1)
    invmap1, invmap2 = inverse_mapping(xmap,ymap)
    
    dst2 = cv2.remap(img2_orig, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)#[:,205:-155,:]
    #cv2.imwrite("/home/leon/Experiments/output/endomapper/SEP2021/dist44.png",dst2)
    matched_pts2 = cv2.KeyPoint_convert(list(np.array(kp2scaled)))
    und2 = np.float32(np.zeros(matched_pts2.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts2[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und2,P=new_K)#new_K instead of the second mtx
    undi2=[]
    for i in range(matched_pts2.shape[0]):
        x_dist=int(matched_pts2[i,0]+205)
        y_dist=int(matched_pts2[i,1])
        undi2.append(cv2.KeyPoint(int(invmap1[y_dist,x_dist]),int(invmap2[y_dist,x_dist]),int(matched_pts2[i].size)))#int(unds[i,0]),int(unds[i,1])'''


    '''dst1 = cv2.remap(img1_orig, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)#[:,205:-155,:]
    #cv2.imwrite("/home/leon/Experiments/output/endomapper/SEP2021/dist44.png",dst1)
    matched_pts1 = cv2.KeyPoint_convert(list(np.array(m_kp1)))
    und1 = np.float32(np.zeros(matched_pts1.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts1[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und1,P=new_K)
    undi1=[]
    for i in range(matched_pts1.shape[0]):
        x_dist=int(matched_pts1[i,0]+205)
        y_dist=int(matched_pts1[i,1])
        undi1.append(cv2.KeyPoint(int(invmap1[y_dist,x_dist]),int(invmap2[y_dist,x_dist]),int(matched_pts1[i].size)))#int(unds[i,0]),int(unds[i,1])
        
    # We select only inlier points
    pts1 = matched_pts1[:,np.newaxis,:]#cv2.KeyPoint_convert(undi1)#und1[:,0,:]#
    pts2 = matched_pts2[:,np.newaxis,:]#cv2.KeyPoint_convert(undi2)#und2[:,0,:]#
        
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,Fc)
    lines1 = lines1.reshape(-1,3)
    good1 = distanceCheck(img1_orig,lines1,pts1,pts2,5)
    print(np.sum(good1))
    img5,img6 = drawlines(img1_orig,img2_orig,lines1[good1==1],pts1[good1==1],pts2[good1==1])
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,Fc)
    lines2 = lines2.reshape(-1,3)
    good2 = distanceCheck(img2_orig,lines2,pts2,pts1,5)
    print(np.sum(good2))
    img3,img4 = drawlines(img2_orig,img1_orig,lines2[good2==1],pts2[good2==1],pts1[good2==1])
    if output is not None:
        cv2.imwrite('/home/leon/Experiments/output/endomapper/MIDL2022/epipoles/'+output, img3)
    plt.figure(5),plt.imshow(img5)
    plt.savefig('/home/leon/Experiments/output/non5.png')
    plt.figure(3),plt.imshow(img3)
    plt.savefig('/home/leon/Experiments/output/non3.png')
    #cv2.waitKey(0)'''
    
    # Draw matches
    print(len(matches))
    matches = np.array(matches)[np.array(inliers_E).astype(bool)].tolist()
    print(len(matches))
    #matches=[]
    '''err_q1, err_t1 = evaluate_R_t(dR, dT, R_E, t_E)
    err_q2, err_t2 = evaluate_R_t(dR, dT, R_E2, t_E)
    if err_q1 <= err_q2:
        err_q = err_q1
        err_t = err_t1
    else:
        err_q = err_q2
        err_t = err_t2
    print(dR,dT)
    print(len(matches), np.rad2deg(err_q1), np.rad2deg(err_q2), err_t1, err_t2, np.rad2deg(err_q), err_t)
    #print(dR, R_E, R_E2, dT, t_E)'''
    matched_img = cv2.drawMatches(#cv2.resize(img1_color, (256,256)), kp1scaled,
                                  img1_color, kp1u,
                                  # cv2.resize(img2_color, (256,256)), kp2scaled,
                                  #cv2.resize(dst2, (int(1440*256/1080),256)), undi2,
                                  #dst2, undi2,
                                  img2_color, kp2u,
                                  matches, None,
                                  matchColor=(0, 255., 0.),
                                  singlePointColor=(0., 0., 255.),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )#NOT_DRAW_SINGLE_POINTS)#
    if output is not None:
        cv2.imwrite('/home/leon/Experiments/output/endomapper/MICCAI2022/workshop/'+output, matched_img)
    cv2.imshow("Matches", matched_img)

    cv2.waitKey(0)
