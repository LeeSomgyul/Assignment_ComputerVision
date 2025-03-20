import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib

#한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')

#원본 형식으로 이미지 가져오기
img = cv.imread('insideout.png', cv.IMREAD_UNCHANGED)

#[이진화]
#이미지를 그레이스케일 처리(하얀색~회색~검정 등)
if len(img.shape) == 3 and img.shape[2] == 4:
    img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
else:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#이진화(완전한 흰색 OR 검정색으로 변환환)
t, bin_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


#[잘라낸 영상]
b = bin_img[bin_img.shape[0]//2 : bin_img.shape[0], 0 : bin_img.shape[0]//2+1]

#[팽창, 침식, 닫기]
#팽창, 침식, 닫기를 위한 필터
se = np.uint8([[1,1,1,1,1],
               [1,1,1,1,1], 
               [1,1,1,1,1], 
               [1,1,1,1,1], 
               [1,1,1,1,1]])

#1.팽창
b_dilation = cv.dilate(b, se, iterations=1)

#2.침식
b_erosion = cv.erode(b, se, iterations=1)

#3.닫기
b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)


#한 화면에 5개 이미지 한번에 출력
plt.figure(figsize=(15, 10))

#첫번째: [이진화]이미지
plt.subplot(2, 3, 1)
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.title("이진화 이미지")

#두번째: [잘라낸 영상]이미지
plt.subplot(2, 3, 2)
plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.title("잘라낸 영상")

#세번째: [팽창]이미지
plt.subplot(2, 3, 3)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.title("팽창")

#네번째: [침식]이미지
plt.subplot(2, 3, 4)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.title("침식")

#다섯번째: [닫기]이미지
plt.subplot(2, 3, 5)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.title("닫기")

#동시 출력
plt.show()