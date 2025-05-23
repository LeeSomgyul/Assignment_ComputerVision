import cv2 as cv;

img = cv.imread('insideout.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#1. 그레이 이미지에 블러 처리 해보기
blur = cv.GaussianBlur(gray, (5, 5), 0)

#2. 임계값 변경 + 블러 이미지 추가 
gray_canny1 = cv.Canny(gray, 10, 50)
gray_canny2 = cv.Canny(gray, 200, 300)
blur_canny = cv.Canny(blur, 50, 150)

cv.imshow('gray_canny1', gray_canny1)
cv.imshow('gray_canny2', gray_canny2)
cv.imshow('blur_canny', blur_canny)

cv.waitKey()
cv.destroyAllWindows()

