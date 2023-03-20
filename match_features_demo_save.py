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
import matplotlib.pyplot as plt

def extract_SIFT_keypoints_and_descriptors(img, m=None, mask=None, feature='sift', green=False):
    if not green:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img[:,:,1]
    if 'orb' in feature or 'akaze' in feature:
        det = cv2.ORB_create(nfeatures=10000)#
    else:
        det = cv2.xfeatures2d.SIFT_create(nfeatures=10000)#
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
    pts=getPtsFromHeatmap(
            semi_thd, conf_thresh=conf_thresh, nms_dist=nms_dist)
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
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches
    

def match_descriptors_cv2_ratio(kp1, desc1, kp2, desc2, feature='sift'):
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
            if m.distance < 0.75*n.distance:
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
    
def distanceCheck(lines,pts1,pts2,thresholdDist=20):
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
    
def compute_inliers_3(matched_kp1, matched_kp2, geom_model='H',video='0',Rot=None,Trans=None):
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
    Es = np.eye(3)
    inliers_E = np.array([0])
    '''cv2.findEssentialMat(und1[:,0, [1, 0]],
                                    und2[:,0, [1, 0]],
                                    mtx,
                                    cv2.RANSAC,
                                    prob = 0.9999,
                                    threshold = 3.)'''
    if inliers_E is None:
        inliers_E = np.array([0])
        #inliers_Ef = np.array([0])
        inliers_Ec = np.array([0])
        R = np.eye(3)
        R2 = np.eye(3)
        t = np.eye(3,1)
    else:
        E = Es[:3,:]
        n = 1
        while np.isnan(np.sum(E)):
            E = Es[n*3:(n+1)*3]
            n += 1
        inliers_Ef = inliers_E.copy()
        #_, _, _, _ = cv2.recoverPose(E, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, mask=inliers_Ef)
        R, R2, t = np.eye(3),np.eye(3),np.eye(3,1)#cv2.decomposeEssentialMat(E)#estimateRotAndTranslFromEssentialMatrix(Ess)#
    
                                        
    Trans_cross = np.array([[0.,-Trans[2,0],Trans[1,0]],
                [Trans[2,0],0.,-Trans[0,0]],
                [-Trans[1,0],Trans[0,0],0.]])
    Ec = Trans_cross @ Rot
    
    # We select only inlier points
    pts1 = und1[:,0,:]#cv2.KeyPoint_convert(undi1)#
    pts2 = und2[:,0,:]#cv2.KeyPoint_convert(undi2)#
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    Fc = np.linalg.inv(mtx.T) @ Ec @ np.linalg.inv(mtx)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,Fc)
    lines2 = lines2.reshape(-1,3)
    #print(lines2.shape, pts2.shape,pts1.shape)
    #inliers_Ec = distanceCheck(lines2,pts2,pts1,3)
    inliers_Ef = distanceCheck(lines2,pts2,pts1,5)
    #print(np.sum(inliers_Ef),np.sum(inliers_Ec))
    inliers_Ec = inliers_Ef.copy()
    _, _, _, _, _ = cv2.recoverPose(Ec, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, distanceThresh=1000., mask=inliers_Ec)
    
    inliers_H = inliers_H.flatten()
    inliers_E = inliers_E.flatten()
    inliers_Ef = inliers_Ef.flatten()
    inliers_Ec = inliers_Ec.flatten()
    return inliers_H, inliers_E, inliers_Ef, inliers_Ec, R, R2, t


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
    Es = np.eye(3)
    inliers_E = np.array([0])
    '''cv2.findEssentialMat(und1[:,0, [1, 0]],
                                    und2[:,0, [1, 0]],
                                    mtx,
                                    cv2.RANSAC,
                                    prob = 0.9999,
                                    threshold = 3.)'''
    if inliers_E is None:
        inliers_E = np.array([0])
        inliers_Ef = np.array([0])
        inliers_Ec = np.array([0])
        R = np.eye(3)
        R2 = np.eye(3)
        t = np.eye(3,1)
    else:
        E = Es[:3,:]
        n = 1
        while np.isnan(np.sum(E)):
            E = Es[n*3:(n+1)*3]
            n += 1
        inliers_Ef = inliers_E.copy()
        #_, _, _, _ = cv2.recoverPose(E, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, mask=inliers_Ef)
        R, R2, t = np.eye(3),np.eye(3),np.eye(3,1)#cv2.decomposeEssentialMat(E)#estimateRotAndTranslFromEssentialMatrix(Ess)#
    
                                        
        Trans_cross = np.array([[0.,-Trans[2,0],Trans[1,0]],
                    [Trans[2,0],0.,-Trans[0,0]],
                    [-Trans[1,0],Trans[0,0],0.]])
        Ec = Trans_cross @ Rot
        inliers_Ec = inliers_H.copy()
        _, _, _, _, _ = cv2.recoverPose(Ec, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, distanceThresh=1000., mask=inliers_Ec)
    
    inliers_H = inliers_H.flatten()
    inliers_E = inliers_E.flatten()
    inliers_Ef = inliers_Ef.flatten()
    inliers_Ec = inliers_Ec.flatten()
    return inliers_H, inliers_E, inliers_Ef, inliers_Ec, R, R2, t


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
    '''if 'E' in geom_model:
        # Estimate the essential matrix between the matches using RANSAC
        T, inliers = cv2.findEssentialMat(und1[:,0, [1, 0]],
                                        und2[:,0, [1, 0]],
                                        mtx,
                                        cv2.RANSAC,
                                        prob = 0.9999,
                                        threshold = 3.)
    elif 'F' in geom_model:
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
        _, R, t, _ = cv2.recoverPose(T, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, mask=inliers)
        inliers = inliers.flatten()'''
    
                                        
    Trans_cross=np.array([[0.,-Trans[2,0],Trans[1,0]],
                [Trans[2,0],0.,-Trans[0,0]],
                [-Trans[1,0],Trans[0,0],0.]])
    Ess=np.matmul(Rot,Trans_cross)
    
    inliers=np.ones((und1[:,0, [1, 0]].shape[0],1),dtype=np.uint8)
    _, R, t, _ = cv2.recoverPose(Ess, und1[:,0, [1, 0]], und2[:,0, [1, 0]], mtx, mask=inliers)
    inliers = inliers.flatten()
    
    return Ess, inliers, R, -np.matmul(R.T,t)


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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('weights_name', type=str)
    parser.add_argument('img1_path', type=str, default='1')
    parser.add_argument('img2_path', type=str, default='2')
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
    parser.add_argument('--nms', type=int, default=1,
                        help='Non-maximum supression size \
                        (default: 1)')
    parser.add_argument('--green', action='store_true', default=False,
                         help='Use only the green channel of the images.')
    parser.add_argument('--spec', action='store_true', default=False,
                         help='Filter out features on top of specularities.')
    args = parser.parse_args()

    weights_name = args.weights_name
    img1_number = args.img1_path
    img2_number = args.img2_path
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

    weights_root_dir = Path(EXPER_PATH)
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)
    
    if 'superpoint' in weights_name or 'sp_v6' in weights_name:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        
        
    if 'sift' in weights_name:
        prefe='sift'
    elif 'orb' in weights_name:
        prefe='orb'
    elif 'spec100pixels' in weights_name:
        prefe='spec'
    elif 'ucluzlabel100' in weights_name:
        prefe='ft'
    elif 'superpoint_v1' in weights_name:
        prefe='super'
    
    filedir='/media/discoGordo/dataset_leon/UZ/MIDL2022'
    videos_seq = {}
    for video in os.listdir(filedir):
        if os.path.isdir(filedir+"/"+video):
            video_path = filedir+"/"+video
            videos_seq[video] = [x for x in os.listdir(video_path) if os.path.isdir(video_path+"/"+x)]
    
    fileouts='/media/discoGordo/dataset_leon/colmap_MIDL/kps'+"/"+prefe
    if not os.path.exists(fileouts):
        os.mkdir(fileouts)
    for v in videos_seq:
        #if "00364" not in v and "00327" not in v:
        #    continue
        if not os.path.exists(fileouts+"/"+v):
            os.mkdir(fileouts+"/"+v)
        for seq in videos_seq[v]:
            #if "9" != seq:
            #    continue
            if not os.path.exists(fileouts+"/"+v+"/"+seq):
                os.mkdir(fileouts+"/"+v+"/"+seq)
            print("Evaluating cluster:", "VIDEO "+v+" CLUSTER "+seq)
            image_files=os.listdir(filedir+"/"+v+"/"+seq)
            for img_file in image_files:
                img1_file=filedir+"/"+v+"/"+seq+"/"+img_file
                
                img1, img1_orig = preprocess_image(img1_file, img_size, green)
                img1_color=img1_orig[:,205:-155]
                
                kp1=None
                desc1=None
                if 'superpoint' in weights_name or 'sp_v6' in weights_name:
                    out1=net(torch.from_numpy(np.expand_dims(img1, 0)).permute(0,3,1,2).to(device))
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
                else:
                    kp1, desc1 = extract_SIFT_keypoints_and_descriptors(img1_color, args.m, mask, weights_name, green)
                if kp1 is None:
                    kp1=np.array([])
                    desc1=np.array([])
                for kp in kp1:
                    x=kp.pt[0]
                    x=x+205
                    y=kp.pt[1]
                    kp.pt=(int(x),int(y))
                kps=cv2.KeyPoint_convert(kp1)
                np.savez_compressed(fileouts+"/"+v+"/"+seq+"/"+img_file[:-4]+'.npz', kps=kps, desc=desc1) 
