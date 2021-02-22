import cv2
import numpy as np
import os, re, glob
from sklearn.model_selection import train_test_split

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255,255,255))

    # return the rotated image
    return rotated

def make_pics():
    groups_folder_path = './datas/'

    categories = ['ga', 'na']
    for i in range(10):
        categories.append(str(i))

    # num_classes = len(categories)

    image_w = 128
    image_h = 128

    X = []
    Y = []

    for idex, categorie in enumerate(categories):
        label = idex
        image_dir = groups_folder_path + categorie + '/'
        # 파일 갯수 만큼
        for top, dir, f in os.walk(image_dir):
            for filename in f:
                print(filename)
                img = cv2.imread(image_dir + filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
                
                # z 각도 조절
                for i in range(-45,46):
                    print('in z i = ' + str(i))
                    img_rot = rotate(img, i)
                    # x 각도 조절
                    for x in range(30):
                        for xx in range(x + 1):
                            # y 각도 조절
                            for y in range(30):
                                bg = cv2.imread('datas/bg.png')
                                # width_bg, height_bg = bg.shape[:2]
                                img_xy = cv2.resize(img_rot, (image_h - y, image_w - x))
                                width_img, height_img = img_xy.shape[:2]
                                for yy in range(y + 1):
                                    bg[xx:width_img + xx, yy:height_img + yy] = img_xy

                                    X.append(bg)
                                    Y.append(label)
            print('len(X) :', len(X), 'len(Y) :', len(Y))
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    xy = (X_train, X_test, Y_train, Y_test)
    
    np.save("./img_data.npy", xy)
    print("DONE")

make_pics()


