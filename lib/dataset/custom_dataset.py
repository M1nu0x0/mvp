from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import os.path as osp
import pickle
import logging
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, get_scale


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


warpAffine = lambda frame, trans, w, h: cv2.warpAffine(frame, trans, (w, h), flags=cv2.INTER_LINEAR)

def multi_thread(func, args):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, *arg) for arg in args]
        return [future.result() for future in futures]


class BaseDataset:
    def __init__(self, dataset_root, seq_list: List[str], cam_list: List[int], transform=None, image_size=(960, 512)):
        self.dataset_root = dataset_root
        self.cam_list = cam_list
        self.num_views = len(cam_list)
        self.transform = transform
        self.image_size = np.array(image_size)    # (width, height)
        self.heatmap_size = self.image_size / 4  # WARNNING!! hard-coded
        
        self.image_set = 'validation'   # WARNNING!! hard-coded
        self._interval = 12  # WARNNING!! hard-coded
        self.seq_list = seq_list
        self.num_seq = len(seq_list)
        self.current_seq = 0
        
        self.db_file = 'group_{}_cam{}.pkl'.\
            format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)
        
        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == seq_list, 'sequence_list mismatch. Please move or delete {}'.format(self.db_file)
            assert info['interval'] == self._interval, 'interval mismatch. Please move or delete {}'.format(self.db_file)
            assert info['cam_list'] == cam_list, 'cam_list mismatch. Please move or delete {}'.format(self.db_file)
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': seq_list,
                'interval': self._interval,
                'cam_list': cam_list,
                'db': self.db,
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)
        
        # dim: (seq, views). (tuple, list)
        self.caps, self.frames, self.fps = zip(*[self._get_caps(seq) for seq in seq_list])
        self.num_frames = [min(frames) for frames in self.frames]
    
    
    def __len__(self):
        return sum(self.num_frames)
    
    
    def __getitem__(self, seq_idx:int, img_idx:int):
        # seq: seq name, idx: frame index
        # --- Inputs ---
        for cap in self.caps[seq_idx]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, img_idx)
        rets, frames = zip(*[cap.read() for cap in self.caps[seq_idx]])
        if not all(rets):
            logger.warning('Some frames are not read correctly')
            return None, None
        height, width = frames[0].shape[:2]
        
        # suggest all frames have the same size
        height, width, _ = frames[0].shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0  # NOTE: do not apply rotation augmentation
        
        trans = get_affine_transform(c, s, r, self.image_size, inv=0)
        # NOTE: this trans represents full image to cropped image,
        # not full image->heatmap
        
        args = [(frame, trans, int(self.image_size[0]), int(self.image_size[1])) for frame in frames]
        inputs = list(multi_thread(warpAffine, args))
        inputs = [self.transform(input) for input in inputs if self.transform]
        
        # --- Meta ---
        aff_trans = np.eye(3, 3)
        aff_trans[0:2] = trans  # full img -> cropped img
        inv_aff_trans = np.eye(3, 3)
        inv_trans = get_affine_transform(c, s, r, self.image_size, inv=1)
        inv_aff_trans[0:2] = inv_trans
        
        # 3x3 data augmentation affine trans (scale rotate=0)
        # NOTE: this transformation contains both heatmap->image scale affine
        # and data augmentation affine
        aug_trans = np.eye(3, 3)
        aug_trans[0:2] = trans  # full img -> cropped img
        hm_scale = self.heatmap_size / self.image_size
        scale_trans = np.eye(3, 3)  # cropped img -> heatmap
        scale_trans[0, 0] = hm_scale[1]
        scale_trans[1, 1] = hm_scale[0]
        aug_trans = scale_trans @ aug_trans
        # NOTE: aug_trans is superset of affine_trans
        
        metas = []
        for view in range(self.num_views):
            camera = self.db[view]['camera']
            cam_intri = np.eye(3, 3)
            cam_intri[0, 0] = float(camera['fx'])
            cam_intri[1, 1] = float(camera['fy'])
            cam_intri[0, 2] = float(camera['cx'])
            cam_intri[1, 2] = float(camera['cy'])
            cam_R = camera['R']
            cam_T = camera['T']
            cam_standard_T = camera['standard_T']
            meta = {
                'center': c,
                'scale': s,
                'rotation': r,
                'camera': camera,
                'camera_Intri': cam_intri,
                'camera_R': cam_R,
                'camera_T': cam_T,
                'camera_standard_T': cam_standard_T,
                'affine_trans': aff_trans,
                'inv_affine_trans': inv_aff_trans,
                'aug_trans': aug_trans,
            }
            metas.append(meta)
        
        return inputs, metas
    
    
    def _get_db(self):
        # panoptic: 프레임 단위 저장, aisl: 시퀀스 단위 저장
        # TODO image 단위로 업데이트 필요
        db = []
        for seq in self.seq_list:
            cameras = self._get_cam(seq)
            for k, v in cameras.items():
                prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                our_cam = dict()
                our_cam['R'] = v['R']
                our_cam['T'] = -np.dot(v['R'].T, v['t']) * 1000.0
                our_cam['standard_T'] = v['t'] * 1000.0
                our_cam['fx'] = np.array(v['K'][0, 0])
                our_cam['fy'] = np.array(v['K'][1, 1])
                our_cam['cx'] = np.array(v['K'][0, 2])
                our_cam['cy'] = np.array(v['K'][1, 2])
                our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)
                
                db.append({
                    'key': "{}_{}".format(seq, prefix),
                    'camera': our_cam,
                })
        return db
    
    
    def _get_caps(self, seq: str) -> Tuple[List[cv2.VideoCapture], List[int], List[float]]:
        caps = [cv2.VideoCapture(osp.join(self.dataset_root, seq, 'hdVideos', 'hd_00_{:02d}.mp4'.format(i))) for i in self.cam_list]
        frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        fps = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
        # width = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in self.caps]
        # height = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in self.caps]
        return caps, frames, fps
    
    
    def _get_cam(self, seq):
        pkl_files = [osp.join(self.dataset_root, seq, 'calibration', 'camera{:d}.pkl'.format(cam)) for cam in self.cam_list]
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
            sel_cam['k'] = sel_cam['distCoef'][[0, 1, 4]].reshape(3, 1)
            sel_cam['p'] = sel_cam['distCoef'][[2, 3]].reshape(2, 1)
            
            # Rotation Matrix (R)
            rvec = np.array(cam['rvec'])
            R, _ = cv2.Rodrigues(rvec)
            sel_cam['R'] = R.dot(M) # Y axis flip
            
            # Translation Vector (t)
            sel_cam['t'] = np.array(cam['tvec']).reshape((3,1))
            sel_cam['standard_T'] = sel_cam['t'] * 1000.0   # mm
            
            # Translation Vector (T) in mm
            sel_cam['T'] = (-np.dot(sel_cam['R'].T, cam['tvec']) * 1000.0)
            
            # fx, fy, cx, cy
            sel_cam['fx'] = sel_cam['K'][0, 0]
            sel_cam['fy'] = sel_cam['K'][1, 1]
            sel_cam['cx'] = sel_cam['K'][0, 2]
            sel_cam['cy'] = sel_cam['K'][1, 2]
            
            # Panel and Node
            panel = 0   # default panel
            node = i+1
            
            # Add to cameras
            cameras[(panel, node)] = sel_cam
        
        return cameras


class AislIter(IterableDataset, BaseDataset):
    def __init__(self, dataset_root, seq=['240705_1300'], cam_list=5, transform=None, start=0, end=None):
        IterableDataset.__init__(self)
        BaseDataset.__init__(self, dataset_root, seq, cam_list, transform)
        self.current_idx = start
        self.end_idx = end
        self.seq_idx = 0
    
    
    def __len__(self):
        return BaseDataset.__len__(self) - self.current_idx
    
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        if self.end_idx is not None and self.current_idx >= self.end_idx:
            raise StopIteration
        
        if self.current_idx >= self.num_frames[self.seq_idx]:
            self.current_idx = self.current_idx % self.num_frames[self.seq_idx]
            self.seq_idx += 1
            if self.seq_idx >= self.num_seq:
                raise StopIteration
        inputs, metas = BaseDataset.__getitem__(self, self.seq_idx, self.current_idx)
        if inputs is None:
            raise StopIteration
        self.current_idx += 1
        return inputs, metas


class Aisl(Dataset, BaseDataset):
    def __init__(self, dataset_root, seq=['240705_1300'], cam_list=5, transform=None):
        Dataset.__init__(self)
        BaseDataset.__init__(self, dataset_root, seq, cam_list, transform)
    
    
    def __len__(self):
        return BaseDataset.__len__(self)


    def __getitem__(self, idx):
        seq_idx = 0
        img_idx = idx
        for num_frames in self.num_frames:
            if idx < num_frames:
                break
            idx -= num_frames
            seq_idx += 1
        for i in range(seq_idx):
            img_idx -= self.num_frames[i]
        inputs, metas = BaseDataset.__getitem__(self, seq_idx, img_idx)
        return inputs, metas
