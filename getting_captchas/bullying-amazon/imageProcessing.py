import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
# from imutils import paths
import pytesseract


def opencv_trials(grey):

    columns = 3
    rows = 5

    ax = plt.subplot(rows, columns, 1)
    ax.set_title("greyed original")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(grey)

    ax = plt.subplot(rows, columns, 2)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ax.set_title("blurred")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(blur)

    ax = plt.subplot(rows, columns, 3)
    bilateral = cv2.bilateralFilter(blur, 5, 75, 75)
    ax.set_title("bilateral")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(bilateral)

    ax = plt.subplot(rows, columns, 4)
    _, thresh = cv2.threshold(bilateral, 150, 255, cv2.THRESH_BINARY)
    ax.set_title("thresholded")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(thresh)

    ax = plt.subplot(rows, columns, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erosion = cv2.erode(thresh, kernel, iterations=1)
    ax.set_title("erosion")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(erosion)

    ax = plt.subplot(rows, columns, 6)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=2)
    ax.set_title("closing")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(closing)

    ax = plt.subplot(rows, columns, 7)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ax.set_title("dist_transform")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(dist_transform)

    ax = plt.subplot(rows, columns, 8)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.02*dist_transform.max(), 255, cv2.THRESH_BINARY)  # ,255,0)
    ax.set_title("sure_fg")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(sure_fg)

    ax = plt.subplot(rows, columns, 9)
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    dilation_1 = cv2.dilate(sure_fg, kernel_1, iterations=2)
    ax.set_title("dilation")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(dilation_1)

    ax = plt.subplot(rows, columns, 10)
    ax.set_title("removing rows")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(extraction(dilation_1))

    ax = plt.subplot(rows, columns, 10)
    ax.set_title("removing rows")
    # ax.set_xticks([])
    # ax.set_yticks([])
    extracted = extraction(dilation_1)
    plt.imshow(extracted)

    ax = plt.subplot(rows, columns, 11)

    # find the contours (continuous blobs of pixels) the image
    _, contours, _ = cv2.findContours(thresh, 1, 2)

    ax.set_title("tagged")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(cv2.drawContours(thresh, contours, -1, (60, 123, 3), 3))

    ax = plt.subplot(rows, columns, rows*columns)
    ax.set_title("original")
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.imshow(img)


def extraction(img):
    # number of black pixels in each row
    row_counts = np.count_nonzero(img == 255, axis=0)
    print(row_counts)
    to_delete = np.argpartition(row_counts, -50)[-50:]
    print(to_delete)
    return np.delete(img, to_delete, axis=0)


if __name__ == "__main__":
    img = cv2.imread("captchas/captcha0.jpg", cv2.IMREAD_GRAYSCALE)
    plt.gray()

    row_counts = np.zeros(img.shape[0]).reshape(img.shape[0], 1)
    column_counts = np.zeros(img.shape[1]).reshape(img.shape[1], 1)
    for file in os.listdir("captchas"):
        img = cv2.imread("captchas/" + file, cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        bilateral = cv2.bilateralFilter(blur, 5, 75, 75)
        _, img = cv2.threshold(bilateral, 150, 255, cv2.THRESH_BINARY)

        img_row_count = np.count_nonzero(img == 0, axis=1)
        img_col_count = np.count_nonzero(img == 0, axis=0)

        img_row_count = img_row_count.reshape(img.shape[0], 1)
        img_col_count = img_col_count.reshape(img.shape[1], 1)

        row_counts += img_row_count
        column_counts += img_col_count

    img = cv2.imread("captchas/captcha0.jpg", cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    bilateral = cv2.bilateralFilter(blur, 5, 75, 75)
    _, img = cv2.threshold(bilateral, 150, 255, cv2.THRESH_BINARY)
    rows_to_delete = np.argsort(row_counts.reshape(1, row_counts.shape[0]))[0]
    cols_to_delete = np.argsort(
        column_counts.reshape(1, column_counts.shape[0]))[0][:80]

    print(cols_to_delete.shape)
    shape = img[:, 0].shape
    for i in range(0, cols_to_delete.__len__()):
        img[:, cols_to_delete[i]] = 255

    new_img = np.delete(img, rows_to_delete[:30], axis=0)[:30]
    plt.imshow(new_img)
    plt.tight_layout()
    plt.show()
