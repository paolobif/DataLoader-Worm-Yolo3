import sys
import cv2
import pandas as pd
sys.path.append("../")
from data_loader import LoadFromCsv

CSV_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/all_data_1_5_21/all_data_1_5_21.csv"
IMG_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/all_data_1_5_21/imgs"

data = LoadFromCsv(CSV_PATH, IMG_PATH)
keys = list(data.csv_dict.keys())

class EfficientLoad(LoadFromCsv):
    """Generator that loads data one image at a time"""
    def __init__(self, csv_path, img_paths):
        super().__init__(csv_path, img_paths)

    def __next__():
        for key in self.csv_dict.keys():
            img_path = os.path.join(self.img_paths, key)
            img = cv2.imread(img_path)

            bbs = self.csv_dict[key]
            for bb in bbs:
                x1, y1 = bb[0], bb[1]
                w, h = bb[2], bb[3]
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(img, (x1,y1), (x2, y2), (255,255,0), thickness=3)

                yield img
