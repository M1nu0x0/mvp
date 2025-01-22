from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import pickle
import logging

import cv2
import numpy as np
from torch.utils.data import Dataset, IterableDataset


logger = logging.getLogger(__name__)

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

class BaseDataset:
    def __init__(self, dataset_root, seq, cam_list, transform=None):
        self.dataset_root = dataset_root
        self.cam_list = cam_list
        self.transform = transform
        
        self.cameras = self._get_cam(seq)
    
    
    def _get_cam(self, seq):
        pkl_files = [osp.join(self.dataset_root, seq, 'calibration', 'camera{}.pkl'.format(cam)) for cam in range(1, self.cam_list+1)]
        cameras = {}
        M = np.array([[1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0]])

        for i, pkl_files in enumerate(pkl_files):
            with open(pkl_files, 'rb') as f:
                cam = pickle.load(f)
            
            sel_cam = {}
            
            # Intrinsic Matrix (K)
            sel_cam['K'] = np.array(cam['camera_matrix'])
            
            # Distortion Coefficients
            sel_cam['distCoef'] = np.array(cam['dist_coeffs']).flatten()
            
            # Rotation Matrix (R)
            rvec = np.array(cam['rvec'])
            R, _ = cv2.Rodrigues(rvec)
            sel_cam['R'] = R.dot(M) # Y axis flip
            
            # Translation Vector (t)
            # T = (-np.dot(sel_cam['R'].T, cam['tvec']) * 1000)
            sel_cam['t'] = np.array(cam['tvec']).reshape((3,1))
            
            # Panel and Node
            panel = 0   # default panel
            node = i+1
            
            # Add to cameras
            cameras[(panel, node)] = sel_cam
        
        return cameras


class AislIter(IterableDataset, BaseDataset):
    def __init__(self, dataset_root, seq='240705_1300', cam_list=5, transform=None, start=0, end=None):
        IterableDataset.__init__(self)
        BaseDataset.__init__(self, dataset_root, seq, cam_list, transform)
        self.start = start
        self.end = end
    
    
    def __iter__(self):
        pass


class Aisl(Dataset, BaseDataset):
    def __init__(self, dataset_root, seq='240705_1300', cam_list=5, transform=None):
        Dataset.__init__(self)
        BaseDataset.__init__(self, dataset_root, seq, cam_list, transform)
    
    
    def __len__(self):
        pass


    def __getitem__(self, idx):
        pass
