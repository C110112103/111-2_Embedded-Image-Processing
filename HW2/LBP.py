import matplotlib.pyplot as plt
import cv2
import numpy as np


def LBP(img):
    dst = np.zeros(img.shape, dtype=img.dtype)
    h, w = img.shape
    start_index = 1
    for i in range(start_index, h-1):
        for j in range(start_index, w-1):
            center = img[i][j]
            code = 0
#             顺时针，左上角开始的8个像素点与中心点比较，大于等于的为1，小于的为0，最后组成8位2进制
            code |= (img[i-1][j-1] >= center) << (np.uint8)(7)
            code |= (img[i-1][j] >= center) << (np.uint8)(6)
            code |= (img[i-1][j+1] >= center) << (np.uint8)(5)
            code |= (img[i][j+1] >= center) << (np.uint8)(4)
            code |= (img[i+1][j+1] >= center) << (np.uint8)(3)
            code |= (img[i+1][j] >= center) << (np.uint8)(2)
            code |= (img[i+1][j-1] >= center) << (np.uint8)(1)
            code |= (img[i][j-1] >= center) << (np.uint8)(0)
            dst[i-start_index][j-start_index] = code
    return dst


# 讀入圖片
image = cv2.imread("1.jpg")
image = cv2.resize(image, (720, 480))

# 圖片轉灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ===========================對整張圖做LBP,並繪製直方圖=========================

# LBP处理
LBP_all = LBP(gray)
all = cv2.calcHist([gray], [0], None, [256], [0, 256])
# 建立新的窗口
fig = plt.figure(figsize=(10, 6))

plt.subplot(234), plt.imshow(LBP_all, 'gray')
plt.subplot(235), plt.imshow(gray, 'gray')
plt.subplot(236), plt.plot(all, color='r')
plt.xlim([0, 256])
plt.show()
# =============================================================================

# 藍色框框
x = 400
y = 330

# 綠色框框
a = 300
b = 330

# 裁切區域的長度與寬度
w = 30
h = 30

# 裁切圖片
gray_cut1 = gray[y:y+h, x:x+w]
gray_cut2 = gray[b:b+h, a:a+w]

img_draw = image

# 標示框選的區域
cv2.rectangle(img_draw, (x, y), (x+w, y+h), (255, 0, 0), 4)
cv2.rectangle(img_draw, (a, b), (a+w, b+h), (0, 255, 0), 4)

cv2.imshow("cut_image", img_draw)
# ================================繪製直方圖====================================

compare1 = cv2.calcHist([gray_cut1], [0], None, [256], [0, 256])
compare2 = cv2.calcHist([gray_cut2], [0], None, [256], [0, 256])
# 建立新的窗口
fig = plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(gray_cut1, 'gray')
plt.subplot(232), plt.imshow(gray_cut2, 'gray')
plt.subplot(234), plt.plot(compare1, color='b')
plt.subplot(235), plt.plot(compare2, color='g')
plt.xlim([0, 256])
plt.show()
# =============================================================================

# 計算圖像直方圖
calculate1, bins = np.histogram(gray_cut1.ravel(), 256, [0, 256])
calculate2, bins = np.histogram(gray_cut2.ravel(), 256, [0, 256])

# 取出現最多的值
# x_axis1 = bins[np.argmax(calculate1)]
# x_axis2 = bins[np.argmax(calculate2)]

# 取得排序後的 index
sorted_idx1 = np.argsort(calculate1)[::-1]
sorted_idx2 = np.argsort(calculate2)[::-1]

# 取出前三個 x 軸的值
top_three_x1 = bins[sorted_idx1][:3]
top_three_x2 = bins[sorted_idx2][:3]

# 取平均
avg1 = sum(top_three_x1) / 3
avg2 = sum(top_three_x2) / 3


print(f"範圍一(藍)出現前三多的值為{top_three_x1},平均值為" + format(avg1, ".2f"))
print(f"範圍二(綠)出現前三多的值為{top_three_x2},平均值為" + format(avg2, ".2f"))

# 平均相減取絕對值
comparison = abs(avg1 - avg2)

# 判斷是否相似
if (comparison > 20):
    print("範圍一(藍)跟範圍二(綠)不相似")
else:
    print("範圍一(藍)跟範圍二(綠)相似")


cv2.imshow('LBP', LBP_all)
cv2.waitKey()
