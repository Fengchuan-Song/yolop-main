import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .AutoDriveDataset import AutoDriveDataset


class WaterScenesDataset(AutoDriveDataset):
    """
    Dataset adapter for WaterScenes.

    Expected labels:
      - detection: YOLO txt, cls cx cy w h, normalized
      - drivable area: semantic mask where class id 8 is drivable area
      - waterline: binary mask where 1 is waterline
    """

    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        print('building WaterScenes database...')
        gt_db = []
        split_file = self._split_file()

        with open(split_file, 'r') as f:
            items = [line.strip() for line in f.readlines() if line.strip()]

        for item in tqdm(items):
            image_path = self._resolve_image_path(item)
            stem = Path(image_path).stem
            label_path = Path(self.cfg.DATASET.LABELROOT) / (stem + '.txt')
            mask_path = Path(self.cfg.DATASET.MASKROOT) / (stem + '.png')
            lane_path = Path(self.cfg.DATASET.LANEROOT) / (stem + '.png')

            gt = self._read_yolo_label(label_path)
            gt_db.append({
                'image': str(image_path),
                'label': gt,
                'mask': str(mask_path),
                'lane': str(lane_path)
            })

        print('WaterScenes database build finish')
        return gt_db

    def _split_file(self):
        split_name = self.cfg.DATASET.TRAIN_SET if self.is_train else self.cfg.DATASET.TEST_SET
        split_path = Path(split_name)
        if not split_path.is_absolute():
            split_path = Path(self.cfg.DATASET.SPLITROOT) / split_path
        if not split_path.exists():
            raise FileNotFoundError('Split file not found: {}'.format(split_path))
        return split_path

    def _resolve_image_path(self, item):
        item_path = Path(item)
        if item_path.is_absolute() and item_path.exists():
            return item_path

        candidate = Path(self.cfg.DATASET.DATAROOT) / item
        if candidate.suffix and candidate.exists():
            return candidate

        stem = item_path.stem if item_path.suffix else item
        return Path(self.cfg.DATASET.DATAROOT) / (stem + '.' + self.cfg.DATASET.DATA_FORMAT)

    @staticmethod
    def _read_yolo_label(label_path):
        if not label_path.exists() or os.path.getsize(label_path) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        labels = np.loadtxt(str(label_path), dtype=np.float32).reshape(-1, 5)
        labels[:, 0] = 0
        labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)
        return labels

    def read_seg_label(self, path):
        label = cv2.imread(path, 0)
        if label is None:
            raise FileNotFoundError('Drivable-area mask not found or unreadable: {}'.format(path))
        return np.where(label == 8, 255, 0).astype(np.uint8)

    def read_lane_label(self, path):
        label = cv2.imread(path, 0)
        if label is None:
            raise FileNotFoundError('Waterline mask not found or unreadable: {}'.format(path))
        return np.where(label == 1, 255, 0).astype(np.uint8)

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        pass
