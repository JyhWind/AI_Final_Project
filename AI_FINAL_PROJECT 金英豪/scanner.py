import cv2
import numpy as np
import rect

# add image here. 在此处添加图像。
# We can also use laptop's webcam if the resolution is good enough to capture 如果分辨率足够高，我们也可以使用笔记本电脑的网络摄像头
# readable document content 可读文档内容
image = cv2.imread('./test.jpg')

# resize image so it can be processed 调整图像大小以便进行处理
# choose optimal dimensions such that important content is not lost 选择最佳尺寸，确保重要内容不会丢失
image = cv2.resize(image, (1500, 880))

# creating copy of original image 创建原始图像的副本
orig = image.copy()

# convert to grayscale and blur to smooth 转换成灰度和模糊到平滑
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#blurred = cv2.medianBlur(gray, 5)

# apply Canny Edge Detection 应用Canny边缘检测
edged = cv2.Canny(blurred, 0, 50)
orig_edged = edged.copy()

# find the contours in the edged image, keeping only the 找到边缘图像中的轮廓，只保留
# largest ones, and initialize the screen contour 最大值，并初始化屏幕轮廓
(contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

#x,y,w,h = cv2.boundingRect(contours[0])
#cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

# get approximate contour 获取近似轮廓
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break


# mapping target points to 800x800 quadrilateral 将目标点映射到800x800四边形
approx = rect.rectify(target)
pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

M = cv2.getPerspectiveTransform(approx,pts2)
dst = cv2.warpPerspective(orig,M,(800,800))

cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


# using thresholding on warped image to get scanned effect (If Required) 对扭曲图像使用阈值以获得扫描效果（如果需要）
ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#cv2.imshow("Original.jpg", orig)
#cv2.imshow("Original Gray.jpg", gray)
#cv2.imshow("Original Blurred.jpg", blurred)
#cv2.imshow("Original Edged.jpg", orig_edged)
cv2.imshow("Outline.jpg", image)
#cv2.imshow("Thresh Binary.jpg", th1)
#cv2.imshow("Thresh mean.jpg", th2)
#cv2.imshow("Thresh gauss.jpg", th3)
#cv2.imshow("Otsu's.jpg", th4)
#cv2.imshow("dst.jpg", dst)

# other thresholding methods
"""
ret,thresh1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(dst,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(dst,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(dst,127,255,cv2.THRESH_TOZERO_INV)

cv2.imshow("Thresh Binary", thresh1)
cv2.imshow("Thresh Binary_INV", thresh2)
cv2.imshow("Thresh Trunch", thresh3)
cv2.imshow("Thresh TOZERO", thresh4)
cv2.imshow("Thresh TOZERO_INV", thresh5)
"""

cv2.waitKey(0)
cv2.destroyAllWindows()

