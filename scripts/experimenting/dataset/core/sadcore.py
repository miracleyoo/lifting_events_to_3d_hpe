"""
Core dataset implementation. BaseCore may be inherhit to create a new
DatasetCore
"""

import os
import yaml
import os.path as op
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from scipy import io

from experimenting.utils import Skeleton

from ..utils import get_file_paths
from .base import BaseCore


class SadCore(BaseCore):
    """
    DHP19 dataset core class. It provides implementation to load frames,
    heatmaps, 2D joints, 3D joints
    """

    MAX_WIDTH = 346
    MAX_HEIGHT = 260
    N_JOINTS = 13
    DEFAULT_TEST_SUBJECTS = [0]
    DEFAULT_TEST_VIEW = [0]
    TORSO_LENGTH = 50 #! Not sure the unit here, using m

    def __init__(
        self,
        name,
        root,
        # data_dir,
        # joints_dir,
        partition,
        n_joints=13,
        n_channels=1,
        test_subjects=None,
        test_cams=None,
        avg_torso_length=TORSO_LENGTH,
        *args,
        **kwargs,
    ):
        super(SadCore, self).__init__(name, partition)
        data_dir = op.join(root, 'frames')
        joints_dir = op.join(root, 'labels')
        config_path = op.join(root, 'config.yaml')
        with open(config_path, 'r') as f:
            data_config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.train_paths, self.test_paths = SadCore._get_file_paths(data_dir, data_config)
        self.file_paths = self.train_paths
        self.file_paths.extend(self.test_paths)

        self.in_shape = (SadCore.MAX_HEIGHT, SadCore.MAX_WIDTH)
        self.n_channels = n_channels
        self.n_joints = n_joints

        self.avg_torso_length = avg_torso_length
        
        self.classification_labels = [
            SadCore.get_label_from_filename(x_path) for x_path in self.file_paths
        ]

        # self.frames_info = [SadCore.get_frame_info(x) for x in self.file_paths]
        self.joints = self._retrieve_data_files(joints_dir) # retrieve content of files

        self.test_subjects = test_subjects
        if test_cams is None:
            self.test_cams = SadCore.DEFAULT_TEST_VIEW
        else:
            self.test_cams = test_cams

    # X
    @staticmethod
    def get_standard_path(subject, session, movement, frame, cam, postfix=""):
        return "S{}_session_{}_mov_{}_frame_{}_cam_{}{}.npy".format(
            subject, session, movement, frame, cam, postfix
        )

    # √
    @staticmethod
    def load_frame(path):
        x = np.load(path, allow_pickle=True) / 255.0
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        return x.astype(float)

    # √
    def get_frame_from_id(self, idx):
        return SadCore.load_frame(self.file_paths[idx])

    # N
    def get_label_from_id(self, idx):
        return self.classification_labels[idx]
    
    # √
    def get_joint_from_id(self, idx):
        joints_file = np.load(self.joints[idx])
        xyz = joints_file["xyz"].swapaxes(0, 1)
        intrinsic_matrix = torch.tensor(joints_file["camera"])
        extrinsic_matrix = torch.tensor(joints_file["M"])
        return Skeleton(xyz), intrinsic_matrix, extrinsic_matrix

    # N
    def get_heatmap_from_id(self, idx):
        hm_path = self.heatmaps[idx]
        return load_heatmap(hm_path, self.N_JOINTS)

    # √
    @staticmethod
    def _get_file_paths(data_dir, data_config):
        def get_paths_part(part='train'):
            settings = data_config[part]
            file_paths = np.array(get_file_paths(data_dir, extensions=[".npy"]))
            # print(data_dir, settings, file_paths[0])

            for setting in settings:
                # print(setting, len(file_paths))
                file_paths = [p for p in file_paths if setting[0] in p and setting[1] in p]
            return file_paths
        
        train_paths = get_paths_part('train')
        test_paths  = get_paths_part('test')
        return train_paths, test_paths

    # X
    @staticmethod
    def get_frame_info(filename):
        filename = os.path.splitext(os.path.basename(filename))[0]

        result = {
            "subject": int(
                filename[filename.find("S") + 1 : filename.find("S") + 4].split("_")[0]
            ),
            "session": int(SadCore._get_info_from_string(filename, "session")),
            "mov": int(SadCore._get_info_from_string(filename, "mov")),
            "cam": int(SadCore._get_info_from_string(filename, "cam")),
            "frame": SadCore._get_info_from_string(filename, "frame"),
        }

        return result

    # √
    def get_test_subjects(self):
        return self.test_subjects

    # √
    def get_test_view(self):
        return self.test_cams

    # X
    @staticmethod
    def _get_info_from_string(filename, info, split_symbol="_"):
        # return int(filename[filename.find(info) :].split(split_symbol)[0].replace(info, ''))
        return int(filename[filename.find(info) :].split(split_symbol)[1])

    # √ Used for classification task, all return 0 here.
    @staticmethod
    def get_label_from_filename(filepath) -> int:
        return 0

    # √
    def _retrieve_data_files(self, labels_dir):
        labels_hm = [
            os.path.join(
                labels_dir, os.path.basename(x).split(".")[0] + '.npz'
            )
            for x in self.file_paths
        ]
        return labels_hm

    # X
    def train_partition_function(self, x):
        return True if x<len(self.train_paths) else False
        # return self.frames_info[x]['subject'] not in self.test_subjects and self.frames_info[x]['cam'] not in self.test_cams
    
    # Test partition_function
    def _partition_function(self, x):
        return True if x>=len(self.train_paths) else False
# N
def load_heatmap(path, n_joints):
    joints = np.load(path)
    h, w = joints.shape
    y = np.zeros((h, w, n_joints))

    for joint_id in range(1, n_joints + 1):
        heatmap = (joints == joint_id).astype('float')
        if heatmap.sum() > 0:
            y[:, :, joint_id - 1] = decay_heatmap(heatmap)

    return y

# √
def decay_heatmap(heatmap, sigma2=10):
    """

    Args
        heatmap :
           WxH matrix to decay
        sigma2 :
             (Default value = 1)

    Returns

        Heatmap obtained by gaussian-blurring the input
    """
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
    heatmap /= np.max(heatmap)  # keep the max to 1
    return heatmap
