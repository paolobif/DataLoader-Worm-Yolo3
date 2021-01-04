import cv2
import csv
import os
import numpy as np
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

class LoadFromCsv():
    """Takes csv and img paths and organizes properties into csv_dict"""
    def __init__(self, csv_path, img_paths):
        self.csv_path = csv_path
        self.img_paths = img_path
        self.img_names = os.listdir(img_path)
        self.csv_dict = {}
        self.dictionize_csv()

    def load_csv(self):
        PATH = self.csv_path
        data = [] #(name, x_uppr, y_uppr, w, h)
        with open(PATH, newline="") as csvfile:
            csvfile = csv.reader(csvfile)
            for file in csvfile:
                if file not in data: #checks for duplicates. this step is slow.
                # quick fix for no_worm in csv
                    if file[1] == "no_worms":
                        pass
                    else:
                        data.append(file)
                else:
                    pass
        return(data)

    def dictionize_csv(self):
        data = self.load_csv()
        for row in data:
            img_name = row[0]
            self.csv_dict.setdefault(img_name, []).append(row[1:])


if __name__ == "__main__":
    TRAIN = True
    CSV_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/compiled_11_20/compiled_11_20.csv"
    IMAGE_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/compiled_11_20/NN_posttrain_2_im/"
    print(f"Getting info from: csv:{CSV_PATH} imgs:{IMAGE_PATH}")


    test = LoadFromCsv(CSV_PATH, IMAGE_PATH)
