import cv2 as cv
import numpy as np

#이미지 불러오기 및 사이즈 조절
img = cv.imread('insideout.png')
img = cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)

#감마를 보정해주는 함수
def gamma(f, gamma = 1.0):
    f1 = f/255.0
    return np.uint8(255*(f1**gamma))

#gamma함수에 매개변수 넣기
gc = np.hstack((gamma(img, 0), gamma(img, 1.0), gamma(img, 100.0)))

#이미지 표시 및 종료
cv.imshow('gamma', gc)
cv.waitKey()
cv.destroyAllWindows

