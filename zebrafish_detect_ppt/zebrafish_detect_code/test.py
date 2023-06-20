import cv2
import numpy as np
from MultiThreshold import MultiThres
import csv

# 載入圖片


def show(name, img):
    try:
        cv2.imshow(name, img)
        cv2.waitKey(0)
    except:
        return


def count_gray_values(img, count_rate):
    A, B, C, D, E = 0, 0, 0, 0, 0
    # Calculate the histogram using OpenCV's calcHist function
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Iterate through the histogram and count the number of pixels for each gray level
    for i in range(256):
        count = histogram[i][0]
        if count > 0:
            print("Gray level", i, "count:", count)

        if i == 0:  # 所有五階的數量
            A = count
        if i == 51:
            B = count
        if i == 102:
            C = count
        if i == 153:
            D = count
        if i == 204:
            E = count
    total = A + B + C + D + E
    row = [
        round((A / total) * 100, 1),
        round((B / total) * 100, 1),
        round((C / total) * 100, 1),
        round((D / total) * 100, 1),
        round((E / total) * 100, 1),
    ]
    count_rate.append(row)

    return count_rate


def ROI(image, multi4Img, multi10Img):
    x_l, x_r, y_u, y_d = 0, 0, 0, 0

    # init
    if len(image.shape) != 2:  # 如果圖像不是灰度圖像，將其轉換為灰階圖像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.medianBlur(image, 3)
    ret, th1 = cv2.threshold(image, 101, 255, cv2.THRESH_BINARY)
    th3 = th1.copy()
    show("th1", th1)
    for y in range(0, image.shape[0], 1):
        # 取得該列中所有非零像素的座標點
        non_zero_pts = cv2.findNonZero(th1[y, :])

        # 如果存在非零像素，則該列中第一個非零像素的x座標即為所求
        if non_zero_pts is not None:
            print("The y-coordinate of the first non-zero pixel from the right is:", y)
            y_u = y
            break

    x_r = find_x_right(th1)

    ret, th2 = cv2.threshold(image, 229, 255, cv2.THRESH_BINARY)

    for x in range(0, image.shape[1], 1):
        # 取得該列中所有非零像素的座標點
        non_zero_pts = cv2.findNonZero(th2[:, x])

        # 如果存在非零像素，則該列中第一個非零像素的x座標即為所求
        if non_zero_pts is not None:
            print("The x-coordinate of the first non-zero pixel from the right is:", x)
            x_l = x + 10
            break

    y_d = find_y_down(th3, x_l)

    ROI4Img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    ROI10Img = ROI4Img.copy()
    # 获取原始图像中的感兴趣区域
    # region_of_interest = image.crop((x_l, y_u, x_r, y_d))

    # 将感兴趣区域复制到新图像中的相应位置
    ROI4Img[y_u:y_d, x_l:x_r] = multi4Img[y_u:y_d, x_l:x_r]
    ROI10Img[y_u:y_d, x_l:x_r] = multi10Img[y_u:y_d, x_l:x_r]

    return ROI4Img, ROI10Img


def find_x_right(th1):  # 找脊椎區域右側
    kernel1 = np.ones((20, 30), np.uint8)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel1)
    (cnts, _) = cv2.findContours(  # cnts是輪廓陣列
        th1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area_max, area_current = 0, 0
    for cnt in cnts:
        (x1, y1, w, h) = cv2.boundingRect(cnt)  # 當前的輪廓
        area = cv2.contourArea(cnt)  # 計算面積

        area_max = max(area_current, area)
        if area_max == area:
            x_r = x1 + w
        area_current = area_max

    return x_r


def find_y_down(th3, x_l):  # 找脊椎區域下方
    th3[:, 0:x_l] = 0
    (cnts, _) = cv2.findContours(  # cnts是輪廓陣列
        th3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area_max, area_current = 0, 0
    for cnt in cnts:
        (x1, y1, w, h) = cv2.boundingRect(cnt)  # 當前的輪廓
        area = cv2.contourArea(cnt)  # 計算面積

        area_max = max(area_current, area)
        if area_max == area:
            y_d = y1
        area_current = area_max

    return y_d


def load_img(img_name):
    # origin image
    origin_img = cv2.imread(img_name)
    origin_img = cv2.resize(origin_img, (920, 540))  # resize

    # gray image
    gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    show("gray_image", gray_img)
    return origin_img, gray_img


def MultiThreshold(gray_img, level):
    maskROI = np.full((gray_img.shape[0], gray_img.shape[1]), 255, dtype=np.uint8)
    multiThreshold_level = MultiThres(gray_img, maskROI, level, 0, 255)
    multiThreshold_level.SearchMax()
    multiThreshold_Img = multiThreshold_level.threshold()
    print(multiThreshold_level.ValueList)
    return multiThreshold_Img


def graySeparate(multiROIImg, graylevel):
    multiROIImgCopy = multiROIImg.copy()
    multiROIImgCopy[multiROIImg != graylevel] = 0
    multiROIImgCopy[multiROIImg == graylevel] = 255
    return multiROIImgCopy


def findContour(multiROIImg128, multiROIImg64):
    kernel = np.ones((3, 3), np.uint8)
    kernel1 = np.ones((20, 1), np.uint8)
    dilateROI128 = cv2.dilate(multiROIImg128, kernel, iterations=1)
    dilateROI128 = cv2.morphologyEx(dilateROI128, cv2.MORPH_CLOSE, kernel)
    dilateROI128 = cv2.morphologyEx(dilateROI128, cv2.MORPH_CLOSE, kernel1)

    dilateROI64 = cv2.dilate(multiROIImg64, kernel, iterations=1)
    close64 = cv2.morphologyEx(dilateROI64, cv2.MORPH_CLOSE, kernel)
    close64 = cv2.morphologyEx(close64, cv2.MORPH_CLOSE, kernel1)
    show("ROIdilate128", dilateROI128)
    show("openingROI64", close64)

    cnts128 = sortContours(dilateROI128)
    cnts64 = sortContours(close64)

    return cnts128, cnts64


def sortContours(multiROIImg):
    (cnts, _) = cv2.findContours(  # cnts是輪廓陣列
        multiROIImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])  # 將cnts排序

    return cnts


def Detect(cnts, fiveThreshold_img, cnts64):
    pre_x = 0  # 上一個脊椎的x
    pre_y = 0  # 上一個脊椎的y
    limit_x = 0
    serial_number = 0  # 標記框選的序列號

    image = origin_img.copy()
    count_rate = []
    for cnt in cnts64:
        area = cv2.contourArea(cnt)  # 計算面積
        if area > 300:
            (x, y, w, h) = cv2.boundingRect(cnt)
            limit_x = x + w
            break
    for cnt in cnts:
        area = cv2.contourArea(cnt)  # 計算面積

        perimeter = cv2.arcLength(cnt, True)  # 計算周長

        (x, y, w, h) = cv2.boundingRect(cnt)  # 當前的輪廓
        if limit_x < x:  # 代表當前的脊椎與上一個脊椎距離過近
            break

        # x = x + w // 2
        # if abs(x - pre_x) < 10:  # 代表當前的脊椎與上一個脊椎距離過近
        #     continue
        # if y - upperbound > 35:  # 代表高度過高，有可能是非脊椎的雜訊
        #     continue
        serial_number = serial_number + 1

        # cv2.rectangle(
        #     image,
        #     (x - 5, upperbound - 5),
        #     (x + 5, y_d),
        #     (0, 0, 255),
        #     2,
        # )  # 框選脊椎
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            1,
        )  # 框選脊椎
        print(
            "Contour # %d - area: %.2f, perimeter: %.2f "
            % (serial_number, area, perimeter)
        )  # 印出結果
        # 5階框框

        pre_x, pre_y = x, y  # 將當前脊椎的資訊儲存起來

        count_rate = count_gray_values(
            fiveThreshold_img[y : y + h, x : x + w], count_rate
        )
        # 在該Contours印上其編號。

        cv2.putText(
            image,
            " %d" % (serial_number),
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    count_rate, image = secondContour(
        cnts64,
        serial_number,
        fiveThreshold_img,
        count_rate,
        image,
        limit_x,
    )
    # return last_x, serial_number, count_rate
    return count_rate, image


def secondContour(
    cnts,
    serial_number,
    fiveThreshold_img,
    count_rate,
    image,
    limit_x,
):  # 找灰階值=64的脊椎
    pre_x = 0  # 上一個脊椎的x
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 當前的輪廓
        if x < limit_x:
            continue

        # if abs(x - pre_x) < 15:  # 代表當前的脊椎與上一個脊椎距離過近
        #     continue
        area = cv2.contourArea(cnt)  # 計算面積
        perimeter = cv2.arcLength(cnt, True)  # 計算周長

        serial_number = serial_number + 1
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            1,
        )  # 框選脊椎
        print(
            "Contour # %d - area: %.2f, perimeter: %.2f "
            % (serial_number, area, perimeter)
        )  # 印出結果
        # 5階框框

        pre_x, pre_y = x, y  # 將當前脊椎的資訊儲存起來

        count_rate = count_gray_values(
            fiveThreshold_img[y : y + h, x : x + w], count_rate
        )
        # 在該Contours印上其編號。

        cv2.putText(
            image,
            " %d" % (serial_number),
            (x - 20, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    return count_rate, image


def creatcsv(count_rate):
    count_rate_with_percentage = [
        [str(data) + " (%)" for data in row] for row in count_rate
    ]  # 將每一個資料後都加上(%)

    with open("data.csv", "w", newline="") as file:
        writer = csv.writer(file)  # 创建 CSV writer 對象
        writer.writerow(
            ["Number", "level1", "level2", "level3", "level4", "level5"]
        )  # 寫入第1行標題

        for i, row in enumerate(count_rate_with_percentage):
            writer.writerow([str(i + 1)] + row)  # 加入編號,在寫入資料


origin_img, gray_img = load_img(
    "20220616_Dex+naringin_with 4X P_2.5X-20230505T092937Z-001\\20220616_Dex+naringin_with 4X P_2.5X\\2uM Dex + GSB\\2uM Dex + GSB -14.jpg"
)  # 11,21,30出問題 12ROI問題
# origin_img, gray_img = load_img("fish7.bmp")
tenThreshold_img = MultiThreshold(gray_img, 10)
show("10multiThreshold", tenThreshold_img)
fiveThreshold_img = MultiThreshold(gray_img, 5)
show("5multiThreshold", fiveThreshold_img)
fourThreshold_img = MultiThreshold(gray_img, 4)
show("4multiThreshold", fourThreshold_img)
multi4ROIImg, multi10ROIImg = ROI(tenThreshold_img, fourThreshold_img, tenThreshold_img)

show("roiimg", multi4ROIImg)

sep4Img = graySeparate(multi4ROIImg, 128)  # 四階是128 10接 102
show("multi128", sep4Img)
sep10Img = graySeparate(multi10ROIImg, 51)  # 四階是128 10接 102
show("test", sep10Img)

# sepImg = graySeparate(tenThreshold_img, 102)  # 四階是128 10接 102
# show("test", sepImg)

cnts4Threshold, cnts10Threshold = findContour(sep4Img, sep10Img)


count_rate, image = Detect(cnts4Threshold, fiveThreshold_img, cnts10Threshold)
show("result", image)

for row in count_rate:
    print(row)

creatcsv(count_rate)
