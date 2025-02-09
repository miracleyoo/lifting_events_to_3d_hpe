import os
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from scipy import io

from experimenting.utils import Skeleton


class BaseCore(ABC):
    """
    Base class for dataset cores. Each core should implement get_frame_info and
    load_frame_from_id for base functionalities. Labels, heatmaps, and joints
    loading may be implemented as well to use the relative task implementations
    """

    def __init__(self, name, partition):
        # self._set_partition_function(partition)
        self.partition = partition
        if hasattr(self, 'n_channels'):
            self._partition_function = None #! Remember to uncomment this line when trained on original dataset!
        self.name = name

    # def _set_partition_function(self, partition_param):
    #     if partition_param is None:
    #         partition_param = "cross-subject"

    #     if partition_param == "cross-subject":
    #         self.partition_function = self.get_cross_subject_partition_function()
    #     else:
    #         self.partition_function = self.get_cross_view_partition_function()

    @property
    def partition_function(self):
        if self._partition_function is not None:
            # print("YEAHHHHHHHHHHHHHHHHHHHH\n\n\n\n")
            return self._partition_function
        else:
            if self.partition is None:
                self._partition_function = self.get_cross_subject_partition_function()
            else:
                self._partition_function = self.get_cross_view_partition_function()
            return self._partition_function

    # def _set_partition_function(self, partition_param):
    #     if partition_param is None:
    #         partition_param = "cross-subject"

    #     if partition_param == "cross-subject":
    #         self.partition_function = self.get_cross_subject_partition_function()
    #     else:
    #         self.partition_function = self.get_cross_view_partition_function()

    @staticmethod
    @abstractmethod
    def get_frame_info(path):
        """
        Get frame attributes given the path

        Args:
          path: frame path

        Returns:
          Frame attributes as a subscriptable object
        """

    def get_cross_subject_partition_function(self):
        """
        Get partition function for cross-subject evaluation method

        Note:
          Core class must implement get_test_subjects
          get_frame_info must provide frame's subject
        """
        def _get(x):
            return self.frames_info[x]["subject"] in self.get_test_subjects()
        # temp = lambda x: self.frames_info[x]["subject"] in self.get_test_subjects()
        return _get

    def get_cross_view_partition_function(self):
        """
        Get partition function for cross-view evaluation method

        Note:
          Core class must implement get_test_view
          get_frame_info must provide frame's cam
        """
        def _get(x):
            return self.frames_info[x]["cam"] in self.get_test_view()
        # temp = lambda x: self.frames_info[x]["cam"] in self.get_test_view()
        return _get

    def train_partition_function(self, x):
        """
        Accept all inputs as training

        """
        return True

    def get_test_subjects(self):
        raise NotImplementedError()

    def get_test_view(self):
        raise NotImplementedError()

    def get_frame_from_id(self, idx):
        raise NotImplementedError()

    def get_label_from_id(self, idx):
        raise NotImplementedError()

    def get_joint_from_id(self, idx) -> Tuple[Skeleton, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_heatmap_from_id(self, idx):
        raise NotImplementedError()
