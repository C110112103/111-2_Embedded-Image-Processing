import cv2
import numpy as np
import time


def process():
    for i in range(1000):
        result = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # open
        # result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations=1)  # close

    return result


if __name__ == "__main__":

    img = cv2.imread('FD6x7QJacAEn88O.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

for i in range(3, 15, 2):  # 執行kernel size 3,5,7,9,11,13

    # --------------------AVX ON-----------------------
    cv2.setUseOptimized(True)

    Start_time_on = time.time()

    kernel = np.ones((i, i), np.uint8)
    result1 = process()

    End_time_on = time.time()

    cv2.putText(result1, "do 1000 times", (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(result1, f"kernal:({i},{i})", (0, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    time_count = (End_time_on - Start_time_on)

    # cv2.imwrite(f'{i}.jpg', result) #存取圖片
    cv2.imshow(f'avx-on{i}x{i}', result1)
    print(f"kernal{i}x{i}")
    print("avx-on 用時{}秒". format(time_count, ".6f"))
    print("")
# --------------------AVX CLOSE-----------------------
    cv2.setUseOptimized(False)

    Start_time_close = time.time()

    kernel = np.ones((i, i), np.uint8)
    result2 = process()

    End_time_close = time.time()

    cv2.putText(result2, "do 1000 times", (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(result2, f"kernal:({i},{i})", (0, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    time_count = (End_time_close - Start_time_close)

    # cv2.imwrite(f'{i}.jpg', result) #存取圖片
    cv2.imshow(f'avx-off{i}x{i}', result2)
    print(f"kernal{i}x{i}")
    print("avx-close 用時{}秒". format(time_count, ".6f"))
    print("")
    time.sleep(10)
# -----------------------------------------------------------
cv2.waitKey()
cv2.destroyAllWindows()
