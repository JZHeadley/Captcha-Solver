import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


def contourImage(img, rows_to_delete, cols_to_delete):
    # blur = cv2.GaussianBlur(img, (3, 3), 0)
    # blur = cv2.medianBlur(img, 3)
    # bilateral = cv2.bilateralFilter(blur, 3, 75, 75)
    # _, thresh = cv2.threshold(bilateral, 125, 255, cv2.THRESH_BINARY)

    new_img = np.delete(img, rows_to_delete[:10], axis=0)  # [:25]
    shape = img[:, 0].shape
    cols_to_delete = cols_to_delete[:15]
    for i in range(0, cols_to_delete.__len__()):
        new_img[:, cols_to_delete[i]] = 0
    kernel = np.ones((4, 2), np.uint8)
    erosion = cv2.erode(new_img, kernel, iterations=1)
    kernel = np.ones((1, 2), np.uint8)
    erosion = cv2.erode(erosion, kernel, iterations=1)

    # dilation=cv2.dilate(erosion,np.ones((1,2),np.uint8),iterations=1)
    new_img = erosion
    _, contours, _ = cv2.findContours(new_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    output = new_img.copy()

    letter_image_regions = []
    for contour in contours:
        # Get the rectangle that contains the contour
        x, y, w, h = cv2.boundingRect(contour)

        # this part here should just black out random flecks of noise that are still left.  
        # in general it probably won't get rid of letter chunks but it could
        if w*h < 15:
            rectContours = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
            cv2.fillPoly(new_img, pts=[rectContours], color=(0, 0, 0))
            continue
        else:
            letter_image_regions.append((x, y, w, h))
            cv2.drawContours(output, contours, -1, 100, 2)

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    for letter_bounding_box in letter_image_regions:
        print(letter_bounding_box)
        
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        cv2.rectangle(output, (x - 2, y - 2),(x + w + 4, y + h + 4), 200, 1)

    return new_img, output


if __name__ == "__main__":
    img = cv2.imread("captchas/captcha0.jpg", cv2.IMREAD_GRAYSCALE)
    # plt.gray()

    row_counts = np.zeros(img.shape[0]-20).reshape(img.shape[0]-20, 1)
    column_counts = np.zeros(img.shape[1]).reshape(img.shape[1], 1)
    count = 1
    captcha_dir = "captchas/"
    out_dir = "captcha_threshs/"
    images = os.listdir(captcha_dir)
    captcha_image_files =images[1:6]
    captcha_image_files = np.random.choice(images, size=(5,), replace=False)
    
    for img_src in captcha_image_files:
        img = cv2.imread(captcha_dir+img_src, cv2.IMREAD_GRAYSCALE)
        or_img=img
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
        columns=3
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

    plt.tight_layout()
    plt.show()
