import cv2
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

DIRECTORY = "./return/416_1_4_full"
CUT_SIZE = 416

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=DIRECTORY, help="path to folder w images and labels")
opt=parser.parse_args()
print(opt)

IMAGE_PATH = f"{opt.path}/images/"
LABEL_PATH = f"{opt.path}/labels/"
img_names = os.listdir(IMAGE_PATH)
img_names = list((i.split(".png")[0] for i in img_names))


class Mini_Frame():
    def __init__(self, name):
        self.name_txt = name + ".txt"
        self.name_img = name + ".png"
        self.bounds = []

        with open((LABEL_PATH+self.name_txt), "r") as text:
            for line in text:
                line = line.strip("\n")
                line = line.split(" ")
                self.bounds.append(line)

    def draw(self):
        im = cv2.imread(IMAGE_PATH+self.name_img)
        y_size, x_size = im.shape[0], im.shape[1]
        for bound in self.bounds:
            print(bound)
            center_x, center_y = float(bound[1])*416, float(bound[2])*416
            bounds_x, bounds_y = float(bound[3])*416 , float(bound[4])*416
            #center_x , center_y = float(bound[1])*x_size, float(bound[2])*y_size
            #bounds_x , bounds_y = float(bound[3])*x_size*0.5, float(bound[4])*y_size*0.5
            upper = (int(center_x - bounds_x), int(center_y - bounds_y)) #(upper left, x,y)
            lower = (int(center_x + bounds_x), int(center_y + bounds_y)) #bottom right (x,y)

            cv2.rectangle(im, upper, lower, (255,0,0), 2)

        cv2.imshow("test", im)
        key = cv2.waitKey(0)
        cv2.destroyWindow("test")




if __name__ == "__main__":
    test = Mini_Frame(img_names[0])

    def cycle_images(number):
        rand_list = np.random.randint(len(img_names), size=number)
        for i in rand_list:
            mini_im = Mini_Frame(img_names[i])
            mini_im.draw()

    cycle_images(100)
