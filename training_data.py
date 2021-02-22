import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def training_data():
    groups_folder_path = './cnn_sample/'

    categories = [str(i) for i in range(10)]
    categories.append('ga')
    categories.append('na')

    num_classes = len(categories)

    image_w = 128
    image_h = 128

    X = []
    Y = []
    
    for idex, categorie in enumerate(categories):
        label = idex
        image_dir = groups_folder_path + categorie
    
        for top, dir, f in os.walk(image_dir):
            for filename in f:
                img = cv2.imread(image_dir+filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
                X.append(img)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    xy = (X_train, X_test, Y_train, Y_test)
    
    np.save("./img_data.npy", xy)
    print("DONE")



