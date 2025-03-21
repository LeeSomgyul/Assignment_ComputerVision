import cv2 as cv
import matplotlib.pyplot as plt

#이미지 불러오기
img = cv.imread('insideout.png')

#명암영상으로 변환 및 히스토그램 출력
#1.평활화 전
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
h_before = cv.calcHist([gray], [0], None, [256], [0,256])

#2.평활화 후
equal = cv.equalizeHist(gray)
h_after=cv.calcHist([equal], [0], None, [256], [0,256])

#한 화면에 결과물 모두 출력
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.plot(h_before, color='r', linewidth=1)

plt.subplot(2, 2, 3)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.plot(h_after, color='r', linewidth=1)

plt.show()