import cv2
import numpy as np
imgwidth = 600
imgheight = 600
cam = cv2.VideoCapture(0)
cam.set(3,imgwidth)
cam.set(4,imgheight)
cam.set(10,150)


def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernal = np.ones((5,5))
    imgDilate = cv2.dilate(imgCanny,kernal,iterations=2)
    imgThres = cv2.erode(imgDilate,kernal,iterations=1)

    return imgThres

def getContoures(img):
    maxArea = 0
    biggest = np.array([])
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imgContour, cnt,-1,(255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest


while True:
    status, img = cam.read()
    img = cv2.resize(img,(imgwidth,imgheight))
    imgContour = img.copy()
    imgThres = preProcessing(img)
    getContoures(imgThres)
    cv2.imshow('doc',imgContour)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
