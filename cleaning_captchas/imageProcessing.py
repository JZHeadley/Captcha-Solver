import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


count = 1
numImages = -1
captcha_dir = "../data/captchas_solved/"
out_dir = "../data/solution_cleaned/"


def deletionMethod(img, row_counts, column_counts):
    global count
    or_img = img
    # got better results with median blurring
    blur = cv2.medianBlur(img, 3)
    bilateral = cv2.bilateralFilter(blur, 3, 75, 75)
    _, img = cv2.threshold(bilateral, 150, 255, cv2.THRESH_BINARY_INV)
    img = img[:-20, :]

    img_row_count = np.count_nonzero(img == 255, axis=1)
    img_col_count = np.count_nonzero(img == 255, axis=0)

    img_row_count = img_row_count.reshape(img.shape[0], 1)
    img_col_count = img_col_count.reshape(img.shape[1], 1)

    row_counts += img_row_count
    column_counts += img_col_count

    rows_to_delete = np.argsort(
        row_counts.reshape(1, row_counts.shape[0]))[0]
    cols_to_delete = np.argsort(column_counts.reshape(
        1, column_counts.shape[0]))[0][:80]
    cleaned, output = contourImage(img, rows_to_delete, cols_to_delete)
    canny = cv2.Canny(cleaned, 50, 150)

    # lines = cv2.HoughLinesP(canny, 1, np.pi/180, 15, np.array([]),
    #                         50, 10)
    # line_img = cleaned.copy()
    # if lines is not None:
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             cv2.line(line_img, (x1, y1), (x2, y2), 150, 1)
    if numImages == -1:
        columns = 3
        cv2.imwrite(out_dir+img_src, cleaned)
        plt.subplot(5, columns, count)
        plt.imshow(or_img)
        count += 1
        plt.subplot(5, columns, count)
        plt.imshow(cleaned)
        count += 1
        # plt.subplot(5, columns, count)
        # plt.imshow(canny)
        # count += 1
        plt.subplot(5, columns, count)
        plt.imshow(output)
        count += 1


def contourImage(img, rows_to_delete, cols_to_delete):
    # blur = cv2.GaussianBlur(img, (3, 3), 0)
    # blur = cv2.medianBlur(img, 3)
    # bilateral = cv2.bilateralFilter(blur, 3, 75, 75)
    # _, thresh = cv2.threshold(bilateral, 125, 255, cv2.THRESH_BINARY)

    new_img = np.delete(img, rows_to_delete[:10], axis=0)  # [:25]
    shape = img[:, 0].shape
    cols_to_delete = cols_to_delete[:10]
    for i in range(0, cols_to_delete.__len__()):
        new_img[:, cols_to_delete[i]] = 0

    kernel = np.ones((4, 2), np.uint8)
    erosion = cv2.erode(new_img, kernel, iterations=1)
    kernel = np.ones((1, 2), np.uint8)
    erosion = cv2.erode(erosion, kernel, iterations=1)

    # dilation=cv2.dilate(erosion,np.ones((1,2),np.uint8),iterations=1)
    new_img = erosion
    _, contours, _ = cv2.findContours(
        new_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    output = new_img.copy()

    letter_image_regions = []
    for contour in contours:
        # Get the rectangle that contains the contour
        x, y, w, h = cv2.boundingRect(contour)

        # this part here should just black out random flecks of noise that are still left.
        # in general it probably won't get rid of letter chunks but it could
        if w*h < 50:
            rectContours = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
            cv2.fillPoly(new_img, pts=[rectContours], color=0)
            continue
        else:
            letter_image_regions.append((x, y, w, h))
            cv2.drawContours(output, contours, -1, 100, 2)

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    for letter_bounding_box in letter_image_regions:
        print(letter_bounding_box)

        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), 200, 1)

    return new_img, output


def union(a, b):
    # http://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/?answer=90504#post-id-90504
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)


def intersection(a, b):
    # http://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/?answer=90504#post-id-90504
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w < 0 or h < 0:
        return (0, 0, 0, 0)
    return (x, y, w, h)


def join_inner_boxes(letter_image_regions):
    bounding_boxes = []
    for bound_rect1 in letter_image_regions:
        flag = False
        for bound_rect2 in letter_image_regions:
            if bound_rect1 == bound_rect2:
                continue
            bound_rect3 = intersection(bound_rect1, bound_rect2)
            # print(bound_rect3)
            area = bound_rect3[2]*bound_rect3[3]
            if area > 0 and not flag:
                print("found some overlap")
                flag = True
                bounding_boxes.append(union(bound_rect1, bound_rect2))
        if not flag:
            bounding_boxes.append(bound_rect1)
    return bounding_boxes


def erosionDilation(img, img_src):
    global count, numImages, out_dir
    or_img = img
    columns = 3
    img = img[:-17, :]
    img = np.delete(img, range(1, 10), axis=0)

    _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)

    img = cv2.medianBlur(img, 3)
    # img = cv2.bilateralFilter(img, 3, 75, 75)

    kernel = np.ones((2, 3), np.uint8)
    erosion_1 = cv2.erode(img, kernel, iterations=1)
    kernel = np.ones((3, 1), np.uint8)

    erosion_2 = cv2.erode(erosion_1, kernel, iterations=1)

    kernel = np.ones((2, 2), np.uint8)
    kernel[0][1] = 0
    kernel[1][0] = 0
    dilation = cv2.dilate(erosion_2, kernel, iterations=1)

    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.erode(dilation, kernel, iterations=1)

    _, contours, _ = cv2.findContours(
        dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _, output = cv2.threshold(dilation, 125, 255, cv2.THRESH_BINARY_INV)

    letter_image_regions = []

    cv2.imwrite(out_dir+img_src, output)
    _, output = cv2.threshold(output, 125, 255, cv2.THRESH_BINARY_INV)

    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    # threshed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, rect_kernel)
    # _, contours, _ = cv2.findContours( threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the rectangle that contains the contour
        x, y, w, h = cv2.boundingRect(contour)

        # this part here should just black out random flecks of noise that are still left.
        # in general it probably won't get rid of letter chunks but it could
        if w*h < 15:
            rectContours = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
            cv2.fillPoly(output, pts=[rectContours], color=0)
            continue
        else:
            letter_image_regions.append((x, y, w, h))
            cv2.drawContours(output, contours, -1, 100, 1)
    print(letter_image_regions.__len__())

    bounding_boxes = join_inner_boxes(letter_image_regions)
    print(bounding_boxes.__len__())

    # letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    # for letter_bounding_box in letter_image_regions:
    #     # print(letter_bounding_box)
    #     # Grab the coordinates of the letter in the image
    #     x, y, w, h = letter_bounding_box
    #     cv2.rectangle(dilation, (x - 2, y - 2), (x + w + 4, y + h + 4), 200, 1)

    for letter_bounding_box in bounding_boxes:
        # print(letter_bounding_box)
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), 225, 1)
    if numImages == -1:
        plt.subplot(numImages, columns, count)
        plt.imshow(or_img)
        count += 1
        plt.subplot(numImages, columns, count)
        plt.imshow(erosion_2)
        count += 1
        # plt.subplot(5, columns, count)
        # plt.imshow(dilation)
        # count += 1
        plt.subplot(numImages, columns, count)
        plt.imshow(output)
        count += 1


if __name__ == "__main__":
    images = os.listdir(captcha_dir)
    if numImages == -1:
        captcha_image_files = images
    else:
        # captcha_image_files = images[:numImages]
        captcha_image_files = np.random.choice(
            images, size=(numImages,), replace=False)
    img = cv2.imread(captcha_dir+captcha_image_files[0], cv2.IMREAD_GRAYSCALE)
    # plt.gray()

    row_counts = np.zeros(img.shape[0]-20).reshape(img.shape[0]-20, 1)
    column_counts = np.zeros(img.shape[1]).reshape(img.shape[1], 1)

    for img_src in captcha_image_files:
        print(img_src)
        img = cv2.imread(captcha_dir+img_src, cv2.IMREAD_GRAYSCALE)
        # deletionMethod(img,row_counts,column_counts)
        erosionDilation(img, img_src)

    plt.tight_layout()
    plt.show()