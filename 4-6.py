from skimage import segmentation, graph
from skimage.color import label2rgb
import numpy as np
import cv2 as cv
import time

img = cv.imread('insideout.png') 
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

start = time.time()

slic = segmentation.slic(img_rgb, compactness=15, n_segments=500, start_label=1)

g = graph.rag_mean_color(img_rgb, slic, mode='similarity')
ncut = graph.cut_normalized(slic, g)
print(img_rgb.shape, 'InsideOut 영상을 분할하는 데', time.time() - start, '초 소요')

marking = segmentation.mark_boundaries(img_rgb, ncut)
ncut_img = np.uint8(marking * 255)

# 픽셀들의 평균 색을 활용해서 색을 덩어리로 나누기기
avg_color = label2rgb(ncut, img_rgb, kind='avg')
avg_color = np.uint8(avg_color * 255)

cv.imshow('Original', cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR))
cv.imshow('Normalized Cut', cv.cvtColor(ncut_img, cv.COLOR_RGB2BGR))
cv.imshow('Avg Color by Region', cv.cvtColor(avg_color, cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()

