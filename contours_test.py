# -*- coding: utf-8 -*-
import cv2
import numpy as np
from corrosion_expand import corrsion_expand

def contours_test(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
    thresh = corrsion_expand(thresh)
    #cv2.imshow("corrsion",thresh)
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    img_list = []

    count = 0
    for cnt in contours[1]:
        x, y, w, h = cv2.boundingRect(cnt)          #得知检测的坐标跟长宽
        res = img[y:y+h,x:x+w]                      #把敏感区域切出来
        print x,y,w,h

        cv2.imwrite("F:\\pic\\res.jpg", res)

        #做一张纯黑的图
        create_img = np.zeros((h/4*5,h/4*5,3), np.uint8)  #

        #纯黑图片和敏感区域叠加

        cre_shape = create_img.shape
        try:
            create_img[(cre_shape[0]-h)/2:(cre_shape[0]+h)/2,(cre_shape[1]-w)/2:(cre_shape[1]+w)/2] = res
        except:
            continue

        dst = cv2.resize(create_img, (100, 100), interpolation=cv2.INTER_AREA)
        name = ("./resize_%d.bmp"%count)

        cv2.imwrite(name, dst)
        count += 1
        img_list.append(dst)
    print img_list.__len__()

    # 用红色表示有旋转角度的矩形框架
    for cnt in contours[1]:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    count = 0

    return img_list

if __name__ == "__main__":

    corrosion_check("../camera_test/screenshot.bmp")