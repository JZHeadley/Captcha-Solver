import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import datetime

count = 1
numImages = -1
numImages = 5
OUTPUT_FOLDER = "./extracted_letters/"
captcha_dir = "../../data/captchas_solved/"
out_dir = "./out/"
correctSeparations = 0
counts = {}


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
    # cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), 225, 1)

    for bound_rect1 in letter_image_regions:
        x1, y1, w1, h1 = bound_rect1
        bound_rect1_temp = (x1, y1-25, w1, y1 + h1+25)
        bound_rect1_temp = (x1-2, y1-25, w1+2, y1 + h1+25)
        flag = False
        for bound_rect2 in letter_image_regions:
            if bound_rect1 == bound_rect2:
                continue
            x2, y2, w2, h2 = bound_rect2
            bound_rect2_temp = (x2, y2-25, w2, y2 + h2+25)
            bound_rect2_temp = (x2-2, y2-25, w2+2, y2 + h2+25)
            bound_rect3 = intersection(bound_rect1_temp, bound_rect2_temp)
            # print(bound_rect3)
            area = bound_rect3[2]*bound_rect3[3]
            if area > 0 and not flag:
                # print("found some overlap")
                flag = True
                bounding_boxes.append(
                    union(bound_rect1_temp, bound_rect2_temp))
        if not flag:
            bounding_boxes.append(bound_rect1)
    return bounding_boxes


def writeOutLetters(image, letter_image_regions, captcha_correct_text):
    captcha_correct_text = captcha_correct_text.replace(
        '.jpg', '').replace('_duplicate', '')
    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y:y + h, x:x + w]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            print("making the directory for extracted letters")
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=3)
        # from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
        # i=0
        # datagen = ImageDataGenerator(
        #         rotation_range=40,
        #         width_shift_range=0.2,
        #         height_shift_range=0.2,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        #         horizontal_flip=True,
        #         fill_mode='nearest')
        # for batch in datagen.flow(letter_image,batch_size=1,save_to_dir=save_path,save_prefix="blah",save_format='png'):
        #     i+=1
        #     if i > 20:
        #         break
        # # increment the count for the current key
        counts[letter_text] = count + 1


def erosionDilation(img, img_src):
    global count, numImages, out_dir, correctSeparations

    or_img = img
    columns = 3
    img = img[:-17, :]
    img = np.delete(img, range(1, 10), axis=0)

    _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)

    img = cv2.medianBlur(img, 3)
    img = cv2.bilateralFilter(img, 3, 75, 75)

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

    # kernel = np.array([
    #     [1, 0, 0, 0, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 0, 0, 0, 1]
    # ], np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # kernel = np.ones((3, 3), np.uint8)
    # dilation = cv2.dilate(erosion, kernel, iterations=2)

    img2 = dilation

    _, contours, _ = cv2.findContours(
        img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, output = cv2.threshold(img2, 125, 255, cv2.THRESH_BINARY_INV)

    letter_image_regions = []

    cv2.imwrite(out_dir+img_src, output)
    _, output = cv2.threshold(output, 125, 255, cv2.THRESH_BINARY_INV)
    output = np.zeros(output.shape, np.uint8)
    for contour in contours:
        # Get the rectangle that contains the contour
        x, y, w, h = cv2.boundingRect(contour)

        # this part here should just black out random flecks of noise that are still left.
        # in general it probably won't get rid of letter chunks but it could
        if w*h < 200:
            rectContours = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
            cv2.fillPoly(output, pts=[rectContours], color=0)
            continue
        if w / h > 1.5 or w*h > 2000:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            if half_width*h < 200:
                rectContours = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
                cv2.fillPoly(output, pts=[rectContours], color=0)
                continue
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))

        else:
            letter_image_regions.append((x, y, w, h))
    # draw out the filled contours
    cv2.drawContours(output, contours, -1, 100, cv2.FILLED)
    # draw the contour lines over top of that for visualization purposes
    cv2.drawContours(output, contours, -1, 156, 0)
    # cv2.fillPoly(output, pts=[np.array(contours),np.uint8], color=255)

    # print(letter_image_regions.__len__())

    bounding_boxes = join_inner_boxes(letter_image_regions)
    # print(bounding_boxes.__len__())

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    for letter_bounding_box in letter_image_regions:
        # print(letter_bounding_box)
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        # cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), 225, 1)
        cv2.rectangle(output, (x, y), (x + w, y + h), 225, 1)

    if bounding_boxes.__len__() == 6:
        correctSeparations += 1
        writeOutLetters(output, letter_image_regions, img_src)

    if numImages != -1:
        plt.subplot(numImages, columns, count)
        plt.imshow(or_img)
        count += 1
        plt.subplot(numImages, columns, count)
        plt.imshow(img2)
        count += 1
        # plt.subplot(5, columns, count)
        # plt.imshow(dilation)
        # count += 1
        plt.subplot(numImages, columns, count)
        plt.imshow(output)
        count += 1


def count_letter_samples(extracted_dir):
    letter_counts = dict()
    for dir in os.listdir(extracted_dir):
        letter_counts[dir] = len(os.listdir(os.path.join(extracted_dir, dir)))

    for letter in sorted(letter_counts.keys()):
        print("We have", letter_counts[letter], "instances of", letter)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    images = os.listdir(captcha_dir)
    if numImages == -1:
        captcha_image_files = images
    else:
        captcha_image_files = np.random.choice(
            images, size=(numImages,), replace=False)
    # plt.gray()

    for img_src in captcha_image_files:
        # print(img_src)
        img = cv2.imread(captcha_dir+img_src, cv2.IMREAD_GRAYSCALE)
        erosionDilation(img, img_src)
    end = datetime.datetime.now()
    if numImages != -1:
        plt.tight_layout()
        plt.show()
    count_letter_samples(OUTPUT_FOLDER)
    print("We had ", correctSeparations, "correct separations")
    print("The image processing for % i images required % s milliseconds CPU time." %
          (images.__len__(), (end - start_time).total_seconds()*1000))
