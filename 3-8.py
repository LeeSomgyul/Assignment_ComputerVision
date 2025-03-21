import cv2 as cv;

#이미지 불러온 후 원하는 부분만 잘라내기기
img = cv.imread('insideout.png')
patch = img[250:350, 170:270, :]

#이미지 잘라내기
img = cv.rectangle(img, (170,250), (270,350), (255,0,0), 3)

#보간
patch1 = cv. resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)
patch2 = cv. resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)
patch3 = cv. resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)

#출력
cv.imshow('Original', img)
cv.imshow('Resize nearest', patch1)
cv.imshow('Resize bilinear', patch2)
cv.imshow('Resize bicubic', patch3)

cv.waitKey()
cv.destroyAllWindows()