import cv2
import numpy as np

h = 1280
w = 720

##イメージの読み込み
img = cv2.imread('G:/pictures/test.png')
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)

##2値化
img_H, img_S, img_V = cv2.split(img_HSV)
_thre, img_dst = cv2.threshold(img_H, 0, 255, cv2.THRESH_BINARY)
img_dst = cv2.resize(img_dst, (h,w))

element4 = np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8) #4近傍
element8 = np.array([[1,1,1], [1,1,1], [1,1,1]], np.uint8) #8近傍
#画像のオープニング処理#
for i in range(3):
    img_dst = cv2.morphologyEx(img_dst, cv2.MORPH_OPEN, element8)

#画像のクロージング処理#
for i in range(3):
    img_dst = cv2.morphologyEx(img_dst, cv2.MORPH_CLOSE, element8)

cv2.imwrite('mask.png', img_dst)

##輪郭の検出
contours, hierarchy = cv2.findContours(img_dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for i in range(0, len(contours)):
    if len(contours[i]) > 0:

        # remove small objects
        if cv2.contourArea(contours[i]) < 500:
            continue

        rect = contours[i]
        x, y, w, h = cv2.boundingRect(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 10)

# save
img = cv2.resize(img,(h,w))

cv2.imwrite('boundingbox.jpg', img)

img_msk = cv2.imread('mask.png')

#マスク処理#
img = cv2.imread('G:/pictures/test.png',1)
img = cv2.resize(img,(h,w))
img_msk = cv2.resize(img_msk, (h,w))

img_dst = cv2.bitwise_and(img, img_msk)
img_dst = cv2.resize(img_dst, (h,w))
cv2.imwrite('masked.png', img_dst)

cv2.namedWindow('img')
cv2.imshow('img', img_dst)
k = cv2.waitKey(0)

'''
grayed = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
# 輪郭抽出
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            areas.append(approx)
    #cv2.drawContours(img, areas, -1, (0,255,0), 3)
    cv2.imwrite(now + '_3_' + file_name + '_boundingbox_B.jpg', img)
'''