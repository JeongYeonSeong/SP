import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import os, re, glob
import cv2
from sklearn.model_selection import train_test_split

catego = ['people', 'nothing']

def get_face_images(num_of_face):
    video = cv2.VideoCapture(0)
    num_frame = 0

    dir = catego
    index = 0

    while video.isOpened():
        num_frame += 1
        ret, frame = video.read()

        cv2.putText(frame, dir[index], (30, 50), cv2.FONT_ITALIC, 0.5, (255, 255, 0), 1)

        cv2.imshow("get_face", frame)

        input_key = cv2.waitKey(10) & 0xFF 
        if input_key == ord('s'):
            cv2.imwrite("./face_sample/" + dir[index] + "/" + dir[index] + str(num_frame) + ".jpg", frame)
        elif input_key == ord('n'):
            index += 1
            if index == 2:
                cv2.destroyAllWindows()
                break

def training_data(max_num, size):
    print("MAKING TRAINING DATA")
    groups_folder_path = './face_sample/'

    categories = catego

    num_classes = len(categories)

    image_w = size
    image_h = size

    X = []
    Y = []
    
    for idex, categorie in enumerate(categories[0 : max_num]):
        label = idex
        image_dir = groups_folder_path + categorie + "/"
    
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
    print("train Data :", len(X_train), ", test Data :", len(X_test))
    np.save("./face_data.npy", xy)
    print("DONE")

def get_model(argNumOfImages, size, maxSize):
    ap = time.time()
    """
    training_data.py를 이용하여 
    """
    # allow_pickle이 기본적으로 False라서 True로 변환해 주지 않으면 읽지 못함
    train_images, test_images, train_labels, test_labels = np.load('face_data.npy', allow_pickle=True)
    class_names = catego

    train_images = train_images[0 : argNumOfImages]
    train_labels = train_labels[0 : argNumOfImages]

    """
    image 전처리
    기본 0 ~ 255 까지 이지만 모델링 과정에서 필요한 것은 0 ~ 1 의 실수
    """
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 모델링 시작
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(size, size,3)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(maxSize, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 모델 학습 시작
    model.fit(train_images, train_labels, epochs=15)

    video = cv2.VideoCapture(0)

    while video.isOpened():
        ret, img = video.read()

        img = cv2.resize(img, dsize=(size, size))
        img2 = (np.expand_dims(img, 0))
        predictions_single = model.predict(img2)
        label = np.argmax(predictions_single[0])
        img = cv2.resize(img, dsize=(1280, 720))
        cv2.putText(img, class_names[label], (0, 40), cv2.FONT_ITALIC, 1, (255, 255, 0), 1)
        cv2.namedWindow("myWindow", 0)
        cv2.resizeWindow("myWindow", 1920, 1080)
        cv2.imshow("myWindow", img)

        if cv2.waitKey(10) > 0:
            break



    # 가중치 테스트
    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    

get_face_images(2)

training_data(2, 256)
get_model(80, 256, 2)



"""
def get_model_for(times, sizes):
    ap = time.time()
    k = []

    for i in range(sizes, sizes * times + 1, sizes):
        k.append(get_model(i, 256, 4))
    for i, acc in enumerate(k):
        print(i + 1,":",acc)

    print(str(int(time.time() - ap)) + " second")

get_model_for(5, 50)
"""