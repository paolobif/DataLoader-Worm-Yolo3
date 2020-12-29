import cv2
import csv
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def load_csv(PATH):
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

def get_raw_im_names(PATH_TO_IMAGES):
    im_names = os.listdir(PATH_TO_IMAGES)
    im_names = list((i.split(".png")[0] for i in im_names))
    return(im_names)

def get_image_bbs(name, csv_ar, stride):
    image_bbs = []
    name = name + ".png"
    for d in csv_ar:
        if name == d[0]:
                w, h = 0.5 * float(d[3]), 0.5 * float(d[4])
                centerX, centerY = float(d[1]) + w, float(d[2]) + h
                #adjusting for shift after the image crop.
                image_bbs.append([centerX, centerY, w, h])
    return(image_bbs)

#only has the worms w/ bouding boxes fully in frame.
def bound_check(lwr_x, lwr_y, x, y, w, h, stride):
    if (lwr_x) <= (x-w) <= (lwr_x+stride) and (lwr_x) <= (x+w) <= (lwr_x+stride):
        if (lwr_y) <= (y-h) <= (lwr_y+stride) and (lwr_y) <= (y+h) <= (lwr_y+stride):
            return(True)
    return(False)

#includes worms on the edge of the Mini_Frame
def check_bounds(lwr_x, lwr_y, centerX, centerY, stride):
    if lwr_x <= centerX <= lwr_x+stride and lwr_y <= centerY <= lwr_y+stride:
        return True
    else:
        False

def xywh2xyxy(bounds):
    x, y, w, h = bounds
    x1, x2 = x - w, x + w
    y1, y2 = y - h, y + h
    return(x1,x2,y1,y2)

def show_image(im, bbs, im_out=False, jup=False):
    for bb in bbs:
        upr_pt = int(bb[0]-bb[2]), int(bb[1]-bb[3])
        lwr_pt = int(bb[0]+bb[2]), int(bb[1]+bb[3])
        cv2.rectangle(im, upr_pt, lwr_pt, (255,0,0),2)

    if not im_out:
        if not jup:
            cv2.imshow("image", im)
            key = cv2.waitKey(0)
            cv2.destroyWindow("image")
        elif jup:
            plt.imshow(im)

    elif im_out:
        return(im)




class rawImage():
    def __init__(self, image_path, name, stride, out_dir, csv_ar):
        self.image_path = image_path
        self.name = name
        self.stride = stride
        self.out_dir = out_dir
        self.image = (cv2.imread(os.path.join(image_path,name+".png")))
        self.bbs = get_image_bbs(name, csv_ar, stride)

    def make_slices(self):
        stride = self.stride
        height = self.image.shape[0]
        width = self.image.shape[1]
        slices_out = []

        count = 0
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                slice_name = f"{self.name}_{count}"
                count += 1
                out_str = ""

                for bb in self.bbs:
                    xc, yc, w, h = bb
                    if check_bounds(x, y, xc, yc, stride):
                        crop = self.image[y:y+stride, x:x+stride]
                        x_dim = crop.shape[1]
                        y_dim = crop.shape[0]
                        cv2.imwrite(f"{self.out_dir}/images/{slice_name}.png", crop)
                        new_centerX = round((xc-x) * (1/x_dim), 6)
                        new_centerY = round((yc-y) * (1/y_dim), 6)
                        new_w = round(w * (2/y_dim), 6)
                        new_h = round(h * (2/x_dim), 6)
                        out_str = out_str + (f"0 {new_centerX} {new_centerY} {new_w} {new_h}\n")
                        slices_out.append(slice_name)

                if not out_str == "":
                    file_out = open(f"{self.out_dir}/labels/{slice_name}.txt", "w+")
                    file_out.write(out_str)
                    file_out.close

        return(slices_out)
