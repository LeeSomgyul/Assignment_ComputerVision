import cv2 as cv

img = cv.imread('insideout.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

insideout = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150,
                            param2=20, minRadius=50, maxRadius=120)

for i in insideout[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)

cv.imshow('InsideOut detection', img)

cv.waitKey()
cv.destroyAllWindows()


