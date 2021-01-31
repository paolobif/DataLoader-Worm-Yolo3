import cv2
import csv
import os
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

## organizes data from csv and folder of images
class LoadFromCsv():
    """Takes csv and img paths and organizes properties into csv_dict"""
    def __init__(self, csv_path, img_paths):
        self.csv_path = csv_path
        self.img_paths = img_paths
        self.img_names = os.listdir(img_paths)
        self.dictionize_csv()

    def load_csv(self):
        PATH = self.csv_path
        df = pd.read_csv(PATH)
        df = df.drop_duplicates()
        #df = df.reset_index()
        return df

    def get_unique(self):
        df = pd.read_csv(self.csv_path, index=False, header=False)
        unique_names = df['frame'].unique()
        df_dict = {}

        print("Initializing Default Dictionary")
        for name in unique_names:
            df_dict.setdefault(name, [])
        print("Done")
        self.df_dict = df_dict

    def dictionize_csv(self):
        data = self.load_csv()
        csv_dict = {}
        # Organizes data into the CSV dictionary
        for row in tqdm(data.iterrows()):
            row = row[1]
            print(row)
            img_name = row[0]
            print(img_name)
            iclass = row[5]
            val = 1 if iclass == 'deadworm' else 0 ## sets worm class to int value. 0 = alive 1 = dead
            # adds encoded class of the worm`
            ## all bounding boxes from new data format is off
            ## this is adjusted to compensate for that. normally *.append([row[1], row[2]... etc])
            x1, y1 = float(row[1]), float(row[2])
            w, h = float(row[3]), float(row[4])
            sftx = float(w)/2
            sfty = float(h)/2
            csv_dict.setdefault(img_name, []).append([x1+sftx, y1+sfty, w, h, val])
            #self.df_dict[img_name].append([row[1], row[2], row[3], row[4], val])
        self.csv_dict = csv_dict

## creates maping generator objectls
class MapGenerator():
    """ Class generates a map grid with the specified cuts
        Takes an imput of either image or xy tuple"""
    def __init__(self, xy_shape, cut_size):
        self.cut_size = cut_size
        if type(xy_shape) != tuple:
            #print("input is an np img array")
            self.xy_shape = (xy_shape.shape[1], xy_shape.shape[0])
            #self.x_shape = img_xy.shape[1]
        else:
            self.xy_shape = xy_shape
            #self.y_shape = img_xy[1]

        self.paired_grid = []

        self.map_grid = self.generate_complete_map_grid()

    def generate_complete_map_grid(self):
        """Four passes of map_grid need to be run in order
        to cover entire area. Returns list of points for bounds. """
        xy_shape = self.xy_shape
        cut_size = self.cut_size

        shiftx = int(self.cut_size/2)
        shifty = int(self.cut_size/2)

        pass1 = self.map_ar(xy_shape, cut_size, (0,0))
        pass2 = self.map_ar(xy_shape, cut_size, (shiftx, shifty))
        #pass3 = self.map_ar(xy_shape, cut_size, (shiftx, 0))
        #pass4 = self.map_ar(xy_shape, cut_size, (0, shifty))
        self.map_grids = [pass1, pass2]#, pass3, pass4]
        #self.map_grids = [pass1]

        complete_map_grid = []
        for map_grid in self.map_grids:
            paired_corners = self.convert_grid_to_cornerxy(map_grid)
            self.paired_grid.append(paired_corners)
            complete_map_grid.extend(paired_corners)

        return complete_map_grid

        # shiftxy can be modified when calling map_ar to create different arrangements of rectangles

    @staticmethod
    def map_ar(xy_shape, cut_size, shiftxy=(0,0)):
        """creates grid map for slicing - adjust shift to cover needed areas -current method
        is shift=(cut_size/2, cut_size/2) and (0, cut_size/2) and (cut_size, 0/2).
        IMPORTANT - cut_size must be div by 2"""
        x_size = xy_shape[0]
        y_size = xy_shape[1]

        x_shift = shiftxy[0]
        y_shift = shiftxy[1]

        map_ar = []

        # declare ranges
        x_start_range = 0 + x_shift
        y_start_range = 0 + y_shift

        # determines the range of each cut.
        second_pass_shift = (int(cut_size/2), int(cut_size/2))
        x_max_range = x_size + cut_size if shiftxy == (0,0) or shiftxy == second_pass_shift else x_size
        y_max_range = y_size + cut_size if shiftxy == (0,0) or shiftxy == second_pass_shift else y_size

        for x in range(0+x_shift, x_max_range, cut_size):
            for y in range(0+y_shift, y_max_range, cut_size):
                map_ar.append([x,y])

        return map_ar

    @staticmethod
    def convert_grid_to_cornerxy(map_grid):
        """ Takes the series of points and then returns point (x1,y1), (x2,y2) for each rectangle"""
        only_xs = [n[0] for n in map_grid]
        y_slice_count = only_xs.count(only_xs[0])

        paired_corners = []
        ## pair index is the matching corner to point
        for i, point in enumerate(map_grid):
            pair_index = i+y_slice_count+1

            # if the pair index is not the last of each chunck or the last group do ...
            if (i+1) % y_slice_count != 0 and pair_index <= len(map_grid):
                #print(i, pair_index, f"division:{(i+1)%y_slice_count}")
                pair = map_grid[pair_index]
                paired_corners.append([tuple(point), tuple(pair)])
            else:
                pass

        return(paired_corners)

## cuts the images according to map dims
class CutImage(MapGenerator):
    """Processes Single Image - cuts it according to cut_size, set
    training True and itll correctly allocate bounding boxes from bb_list: [x_corner,y_corner,w,h]"""
    def __init__(self, img, cut_size, Training=False, bb_list=None):
        """important property is map_grid from MapGenreator - maps where to cut image"""
        MapGenerator.__init__(self, img, cut_size)
        self.img = img
        self.Training = Training
        self.bb_list = bb_list

    def cut_image(self):
        # cuts image according to map_grid and returns dict with img and keys
        img_cuts = {}
        for xy_pair in self.map_grid:
            x1y1, x2y2 = xy_pair
            x1, y1 = x1y1
            x2, y2 = x2y2

            img_crop = self.img[y1:y2, x1:x2]

            # add padding if the image is not sized correctly
            if img_crop.shape != (self.cut_size, self.cut_size):
                img_crop = self.add_padding_to_square_img(img_crop, self.cut_size)

            bbs_in_frame = self.check_if_bb_in_frame(x1y1, x2y2)
            if len(bbs_in_frame) > 0:
                img_cuts.setdefault("keys", []).append(xy_pair)
                img_cuts.setdefault("imgs", []).append(img_crop)
                img_cuts.setdefault("bbs", []).append(bbs_in_frame)

        return img_cuts

    def check_if_bb_in_frame(self, x1y1, x2y2):
        """Finds bounding boxes within the frame bounds and returns list of bbs"""
        bbs_in_frame = []
        for bb in self.bb_list:
            iclass = bb[4]
            bb = bb[:4] #isolates only bounding boxes, excludes class

            bb = [float(n) for n in bb]
            xcorner, ycorner, w, h = bb
            worm_x1y1, worm_x2y2 = (xcorner, ycorner), (xcorner+w, ycorner+h)

            test = {"worm_x1y1":False, "worm_x2y2":False}
            if self.point_is_within_bounds(worm_x1y1, x1y1, x2y2): test["worm_x1y1"] = True
            if self.point_is_within_bounds(worm_x2y2, x1y1, x2y2): test["worm_x2y2"] = True

            #print(test, bb, x1y1, x2y2)
            ## set only one to be true if you want to include bbs that are partially within frame
            if test["worm_x1y1"] == True and test["worm_x2y2"] == True:
                # adjust for shift in position after cutting
                adj_x = xcorner - x1y1[0]
                adj_y = ycorner - x1y1[1]
                bbs_in_frame.append([adj_x, adj_y, w, h, iclass])
                #print(bb, test)

        return bbs_in_frame

    @staticmethod
    def add_padding_to_square_img(img, cut_size):
        y_size, x_size = img.shape[:2]
        y_pad_amount = cut_size - y_size
        x_pad_amount = cut_size - x_size

        pad_img = np.pad(img, [(0,y_pad_amount), (0,x_pad_amount), (0,0)])

        return pad_img

    @staticmethod
    def point_is_within_bounds(worm_xy, x1y1, x2y2):
        #checks if one corner of the worm bb is within the frame
        #if a single point is out of frame the worm is excluded
        if worm_xy[0] >= x1y1[0] and worm_xy[0] <= x2y2[0]:
            if worm_xy[1] >= x1y1[1] and worm_xy[1] <= x2y2[1]:
                return True
        else:
            return False

## Helper functions
def filer_empty_imgs(img_cuts):
    ims = img_cuts["imgs"]
    all_bbs = img_cuts["bbs"]
    new_dict = {}
    for im, bbs in zip(ims, all_bbs):
        if bbs != None:
            new_dict.setdefault("bbs", []).append(bbs)
            new_dict.setdefault("imgs", [].append(im))
    return new_dict


if __name__ == "__main__":
    TRAIN = True ## utilizes already labled boundingboxes
    CUT_SIZE = 416 ## cuts 416x416
    VAL = 0.2 ## proportion of the data that will be allocated to validation set
    DATA_CAP = 15000 ## number of image slices to cap the dataset size to
    IGNORE_EMPTY = True
    CSV_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/AD-data/ad-all-samp.csv"
    IMAGE_PATH = "/home/paolobif/Lab-Work/ml/pre_arch/worm_data/AD-data/AD-raw-samp"
    OUT_NAME = "416_1_27_AD" ## name for the training folder

    print(f"Getting info from: \ncsv:{CSV_PATH} \nimgs:{IMAGE_PATH} \nStoring in ./return as: {OUT_NAME}")

    if not os.path.exists(f"./return/{OUT_NAME}"):
        os.makedirs(f"./return/{OUT_NAME}")
        os.makedirs(f"./return/{OUT_NAME}/labels")
        os.makedirs(f"./return/{OUT_NAME}/images")
    else:
        print("Warning! \nPath already Exists.... Clean up file tree")


    print("Loading CSV")
    data = LoadFromCsv(CSV_PATH, IMAGE_PATH)
    print("CSV Loaded")
    bb_data = data.csv_dict
    img_names = data.img_names

    #cleans up list of images
    images = os.listdir(IMAGE_PATH)
    imgs_with_data = list(bb_data.keys())
    images_to_process = []
    for img_name in images:
        if img_name in imgs_with_data:
            images_to_process.append(img_name)

    random.shuffle(images_to_process)

    ## iterate through all images and find appropriate boxes
    for img_name in tqdm(images_to_process):
        img = cv2.imread(os.path.join(IMAGE_PATH, img_name))
        bb_list = bb_data[img_name]

        ## img_cuts is dict with map bounds, the cut images, and bbs for each image at matching index
        ## keys: 'keys', 'imgs', 'bbs'
        cut_img = CutImage(img, 416, Training=True, bb_list=bb_list)

        # Gets image cuts and filters out empty images if IGNORE_EMPTY == True
        img_cuts = cut_img.cut_image()
        #if IGNORE_EMPTY:
        #    img_cuts = filer_empty_imgs(img_cuts)

        ims = img_cuts["imgs"]
        all_bbs = img_cuts["bbs"]

        ## write image and bbs to folders in labels and images
        img_base_name = os.path.splitext(img_name)[0]
        for i, (im, bbs) in enumerate(zip(ims, all_bbs)):
            new_img_name = f"{img_base_name}_cut{i}"
            new_img_path = os.path.join(f"./return/{OUT_NAME}/images", f"{new_img_name}.png")
            new_label_path  = os.path.join(f"./return/{OUT_NAME}/labels", f"{new_img_name}.txt")

            cv2.imwrite(new_img_path, im)
            ## write labels to text
            with open(new_label_path, "a+") as label_file:
                for bb in bbs:
                    ## scale bounding boxes to 0-1 and convert to center xy w h
                    # cv2.imwrite(new_img_path, im)

                    w, h = (bb[2] / CUT_SIZE), (bb[3] / CUT_SIZE)
                    xcenter = (bb[0] + (0.5*w)) / CUT_SIZE
                    ycenter = (bb[1] + (0.5*h)) / CUT_SIZE

                    ## 0 for worm. 1 for deadworm.
                    iclass = bb[4]
                    # to avoid having center values outside [0,1] bounds
                    if 1 > xcenter > 0  or 1 > ycenter > 0:
                        out_str = f"{iclass} {xcenter} {ycenter} {w} {h}\n"
                        label_file.write(out_str)

            # allows restriction of dataset size to DATA_CAP
            number_of_ims = len(ims)
            if number_of_ims > DATA_CAP:
                number_of_ims = DATA_CAP

            val_count = 0.2 * number_of_ims

            if i < val_count:
                ## write path location for val file
                with open(f"./return/{OUT_NAME}/valid.txt", "a+") as val_file:
                    val_string = (f"{OUT_NAME}/images/{new_img_name}.png\n")
                    val_file.write(val_string)

            elif i < number_of_ims and i > val_count:   # makes sure training data adheres to data cap
                with open(f"./return/{OUT_NAME}/train.txt", "a+") as train_file:
                    train_string = (f"{OUT_NAME}/images/{new_img_name}.png\n")
                    train_file.write(train_string)

    ## make last necessary files
    with open(f"./return/{OUT_NAME}/classes.names", "a+") as f:
        f.write("worm\n")

    with open(f"./return/{OUT_NAME}/{OUT_NAME}.data", "a+") as f:
        f.write(f"classes= 1\ntrain={OUT_NAME}/train.txt\nvalid={OUT_NAME}/valid.txt\nnames={OUT_NAME}/classes.names\n")
