import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


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


def contourImage(img, rows_to_delete, cols_to_delete):
    # blur = cv2.GaussianBlur(img, (3, 3), 0)
    blur = cv2.medianBlur(img, 3)
    bilateral = cv2.bilateralFilter(blur, 3, 75, 75)
    _, thresh = cv2.threshold(bilateral, 125, 255, cv2.THRESH_BINARY_INV)

    new_img = np.delete(thresh, rows_to_delete[:10], axis=0)  # [:25]
    shape = img[:, 0].shape
    cols_to_delete=cols_to_delete[:10]
    for i in range(0, cols_to_delete.__len__()):
        new_img[:, cols_to_delete[i]] = 0
    kernel=np.ones((4,2),np.uint8)
    erosion=cv2.erode(new_img,kernel,iterations=1)
    kernel=np.ones((2,1),np.uint8)
    erosion=cv2.erode(erosion,kernel,iterations=1)
    new_img=erosion
    _, contours, _ = cv2.findContours(
        new_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        x, y, w, h = cv2.boundingRect(contour)

        # if w / h > 1.25:
        #     # This contour is too wide to be a single letter!
        #     # Split it in half into two letter regions!
        #     half_width = int(w / 2)
        #     letter_image_regions.append((x, y, half_width, h))
        #     letter_image_regions.append((x + half_width, y, half_width, h))
        # else:
        #     # This is a normal letter by itself
        letter_image_regions.append((x, y, w, h))
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    output = new_img.copy()
    for letter_bounding_box in letter_image_regions:
        print(letter_bounding_box)
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        cv2.rectangle(output, (x - 2, y - 2),
                      (x + w + 4, y + h + 4), (60, 70, 0), 1)
    return thresh, output


if __name__ == "__main__":
    img = cv2.imread("captchas/captcha0.jpg", cv2.IMREAD_GRAYSCALE)
    # plt.gray()

    row_counts = np.zeros(img.shape[0]-20).reshape(img.shape[0]-20, 1)
    column_counts = np.zeros(img.shape[1]).reshape(img.shape[1], 1)
    count = 1
    captcha_dir = "captchas/"
    out_dir = "captcha_threshs/"
    images = os.listdir(captcha_dir)
    captcha_image_files = np.random.choice(images, size=(5,), replace=False)

    for img_src in captcha_image_files:
        img = cv2.imread(captcha_dir+img_src, cv2.IMREAD_GRAYSCALE)
        # blur = cv2.GaussianBlur(img, (3, 3), 0)
        # got better results with median blurring
        blur = cv2.medianBlur(img, 3)
        bilateral = cv2.bilateralFilter(blur, 3, 75, 75)
        _, img = cv2.threshold(bilateral, 150, 255, cv2.THRESH_BINARY)
        img = img[:-20, :]

        img_row_count = np.count_nonzero(img == 0, axis=1)
        img_col_count = np.count_nonzero(img == 0, axis=0)

        img_row_count = img_row_count.reshape(img.shape[0], 1)
        img_col_count = img_col_count.reshape(img.shape[1], 1)

        row_counts += img_row_count
        column_counts += img_col_count

        rows_to_delete = np.argsort(
            row_counts.reshape(1, row_counts.shape[0]))[0]
        cols_to_delete = np.argsort(column_counts.reshape(
            1, column_counts.shape[0]))[0][:80]
        thresh, output = contourImage(img, rows_to_delete, cols_to_delete)
        canny = cv2.Canny(thresh, 50, 150)

        # lines = cv2.HoughLinesP(canny, 1, np.pi/60, 15, np.array([]),
        #                         50, 20)
        # line_img = img.copy()
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)

        cv2.imwrite(out_dir+img_src, thresh)
        plt.subplot(5, 3, count)
        plt.imshow(img)
        count += 1
        plt.subplot(5, 3, count)
        plt.imshow(thresh)
        count += 1
        plt.subplot(5, 3, count)
        plt.imshow(output)
        count += 1

    plt.tight_layout()
    plt.show()
