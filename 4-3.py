import cv2 as cv
import numpy as np

img = cv.imread('insideout.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 100, 200)
canvas = np.zeros_like(img)

contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

lcontour = []
for i in range(len(contour)):
    if contour[i].shape[0]>100:
        lcontour.append(contour[i])

# 선 색을 파랑색, 두께는 2로 변경해보기 
cv.drawContours(img, lcontour, -1, (255, 0, 0), 2)
# 선만 따로 화면에 보여주기 
cv.drawContours(canvas, lcontour, -1, (0, 255, 255), 1)

cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)
cv.imshow('lind', canvas)

cv.waitKey()
cv.destroyAllWindows()

