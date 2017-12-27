import cv2
import numpy as np

class grapify_test():
    def grapify(self,image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    def thresholding_inv(self,gray):

        ret, bin = cv2.threshold(gray, 48, 255, cv2.THRESH_BINARY_INV)
        bin = cv2.medianBlur(bin, 3)

        return bin

    def grapify_thresholding(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, bin = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        bin = cv2.medianBlur(bin, 3)

        return bin

if __name__ == "__main__":
    img = cv2.imread("screenshot.bmp")
    grapify = grapify_test()
    grap = grapify.grapify(img)
    cv2.imshow("grap",grap)
    cv2.waitKey(0)
    thr = grapify.thresholding_inv(grap)
    cv2.imshow("threshold",thr)
    cv2.imwrite("./threshold/threshold_01.bmp",thr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()