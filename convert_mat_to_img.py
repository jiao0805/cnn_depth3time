#encoding: utf-8
import os
import numpy as np
import h5py
from PIL import Image
import random
import glob

def cvs_test(test_path):
    file_list_rgb=[]
    file_list_range=[]
    trains=[]
    #file_glob_rgb = os.path.join(test_path, '*' + 'colors.png')
    #file_glob_range = os.path.join(test_path, '*' + 'depth.png')
    file_glob_rgb = os.path.join(test_path, '*.' + 'jpg')
    file_glob_range = os.path.join(test_path, '*.' + 'png')
    file_list_rgb.extend(glob.glob(file_glob_rgb))
    file_list_range.extend(glob.glob(file_glob_range))
    # trains = zip(file_list_rgb,file_list_range)
    print(file_list_rgb)
    for rgb, range in zip(file_list_rgb, file_list_range):
        print(rgb)
        trains.append((rgb, range))
    #random.shuffle(trains)

    with open('testforcal.csv', 'w') as output:
        for (image_name, depth_name) in trains:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")


if __name__ == '__main__':
    current_directory = os.getcwd()
    #when train with nyu
    #nyu_train_path = 'data//nyu_datasets'
    #cvs_test(nyu_train_path)
    #nyu_test_path = 'data//nyu_test'
    #cvs_test(nyu_test_path)
    #train_path ='data//train'
    #test_path = 'data//test4'
    inference_path = 'data//testforcal'
    cvs_test(inference_path)
    #cvs_test(train_path)
    #cvs_test(test_path)
    #cvs_test(test_path)

