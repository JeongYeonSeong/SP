import cv2
import numpy as np

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

img = cv2.imread('datas/3/3_1.jpeg')
angle = -5
w,h = img.shape[:2]

img = cv2.resize(img, (128,128))

cv2.imshow('asdf', img)
key = cv2.waitKey(0)

bg = cv2.imread('datas/bg.png')
bw, bh = bg.shape[0:2]

print(bw, bh)

bg[0:, 0:] = img

cv2.imshow('asdf', bg)
key = cv2.waitKey(0)




while 1:
    img_rot = rotate(img, angle)
    pp = 5
    for k in range(10):
        bg = cv2.imread('datas/bg.png')
        bw, bh = bg.shape[0:2]

        img_x = cv2.resize(img_rot, (w - pp, h))
        ww, hh = img_x.shape[0:2]

        for bww in range(0, bw - ww, 50):
            for bhh in range(0, bh - hh, 50):
                bgs = cv2.imread('datas/bg.png')
                bgs[bww:ww + bww, bhh:hh + bhh] = img_x

                cv2.imshow('window', bgs)
        
                key = cv2.waitKey(0)

                if key == ord('q'):
                    break


        pp += 5

        

    key = cv2.waitKey(0)

    if key == ord('p'):
        break
    else:
        angle -= 5