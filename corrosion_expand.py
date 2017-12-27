import cv2

def corrsion_expand(img):
    #img = cv2.imread("../screenshot.bmp")
    img = cv2.erode(img,None,iterations = 1)
    #cv2.imshow("corrrsion",img)
    img = cv2.dilate(img,None,iterations = 14)
    #cv2.imshow("expand",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img