import cv2 as cv;

#이미지 불러온 후 그레이스케일로 변경경
img = cv.imread('insideout.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#소벨 연산자 적용용
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

#소벨 연산자 결과를 화면으로 보여주기(이미지로 표시 가능하도록)
sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)

#위 x, y 엣지를 더하기
edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 50)

cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely', sobel_y)
cv.imshow('edge strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()

