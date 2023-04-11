import cv2 
import numpy as np
import matplotlib.pyplot as plt

global point1, point2, cut_img

def mouse(event, x, y, flags, param):
    global image, point1, point2, cut_img
    img2 = image.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = image[min_y:min_y + height, min_x:min_x + width]



# 讀入圖片
image = cv2.imread("7.jpg")
image = cv2.resize(image, (720, 480))
image_draw  = image.copy()
cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse)
cv2.imshow('img', image)
cv2.waitKey(0)

# 圖片轉灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_cut = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
# ================================繪製直方圖====================================
all = cv2.calcHist([gray], [0], None, [256], [0, 256])
compare = cv2.calcHist([gray_cut], [0], None, [256], [0, 256])
# 建立新的窗口
fig = plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(gray, 'gray')
plt.subplot(232), plt.imshow(gray_cut, 'gray')
plt.subplot(234), plt.plot(compare, color='b')
plt.subplot(235), plt.plot(all, color='r')
plt.xlim([0, 256])
plt.show()
# =============================================================================

# 計算圖像直方圖
calculate_cut, bins = np.histogram(gray_cut.ravel(), 256, [0, 256])


# 取得排序後的 index
sorted_idx = np.argsort(calculate_cut)[::-1]

# 取出前三個 x 軸的值
top_three_x = bins[sorted_idx][:3]

# 取平均
avg_cut = sum(top_three_x) / 3

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        pixel_value = gray[i, j]
        calculate = abs(pixel_value - avg_cut)
        if (calculate) < 30:
            cv2.rectangle(image_draw, (j,i), (j,i), (255,0,0), -1)



cv2.imshow('image_draw', image_draw)
cv2.waitKey()