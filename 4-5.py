import skimage
from skimage.color import label2rgb
import numpy as np
import cv2 as cv

# 내 이미지 가져오기
img = cv.imread('insideout.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 각 슈퍼픽셀을 선이 아닌 영역별로 채우기
slic1 = skimage.segmentation.slic(img, compactness = 20, n_segments = 300)
sp_img1 = label2rgb(slic1, img, kind='avg')
sp_img1 = np.uint8(sp_img1*255.0)

slic2 = skimage.segmentation.slic(img, compactness = 20, n_segments = 900)
sp_img2 = label2rgb(slic2, img, kind='avg')
sp_img2 = np.uint8(sp_img2*255.0)

cv.imshow('Super pixels (compact 20)', cv.cvtColor(sp_img1, cv.COLOR_BGR2RGB))
cv.imshow('Super pixels (compact 40)', cv.cvtColor(sp_img2, cv.COLOR_BGR2RGB))

cv.waitKey()
cv.destroyAllWindows()

