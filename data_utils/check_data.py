import sys
import os
import cv2
import pandas as pd
sys.path.append("../")
from data_loader import LoadFromCsv

CSV_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/all_data_1_5_21/all_data_1_5_21.csv"
IMG_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/all_data_1_5_21/imgs"

class EfficientLoad(LoadFromCsv):
    """Generator that loads data one image at a time"""
    def __init__(self, csv_path, img_paths):
        super().__init__(csv_path, img_paths)
        self.keys = list(self.csv_dict.keys())

        self.loader = self.__next__()

    def __next__(self):
        for key in self.csv_dict.keys():
            img_path = os.path.join(self.img_paths, key)
            img = cv2.imread(img_path)

            try:
                bbs = self.csv_dict[key]
                for bb in bbs:
                    x1, y1 = float(bb[0]), float(bb[1])
                    w, h = float(bb[2]), float(bb[3])
                    x2, y2 = (x1 + w), (y1 + h)
                    cv2.rectangle(img, (int(x1),int(y1)), (int(x2), int(y2)), (255,255,0), 2)
            except:
                print(f"no bbs for {img_path}")

            yield img_path, img

data = EfficientLoad(CSV_PATH, IMG_PATH)



while True:
    path, img = next(data.loader)
    cv2.imshow(path, img)
    key = cv2.waitKey(0)

    if key == ord('s'):
        print("bad", path)
        with open(f"bad.txt", "w+") as txt:
            txt.write(path)

    if key == ord('w'):
        print("good", path)
    if key == ord('d'):
        cv2.destroyAllWindows()

    with open(f"anylized.txt", "w+") as txt:
        txt.write(path)

    cv2.destroyWindow(path)
