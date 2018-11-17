from sklearn.metrics import accuracy_score
import pickle
import cv2
import imutils
import numpy as np
from imutils import paths
from helpers import resize_to_fit
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "../../data/captchas_solved/"
correctSeparations = 0
numChars = 6
true_values = []


def flatten_string_lists(string_list1, string_list2):
    output1 = []
    output2 = []
    for i in range(0, len(string_list1)):
        if len(string_list1[i]) != len(string_list2[i]):
            continue
        for j in range(0, len(string_list1[i])):
            output1.append(string_list1[i][j])
            output2.append(string_list2[i][j])
    return output1, output2


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


def erosionDilation(img, img_src):
    global count, numImages, out_dir, correctSeparations


    or_img = img
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
                rectContours = np.array(
                    [[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
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

    bounding_boxes = join_inner_boxes(letter_image_regions)

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        # cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), 225, 1)
        cv2.rectangle(output, (x, y), (x + w, y + h), 225, 1)

    # bounding_boxes= sorted(bounding_boxes, key=lambda x: x[0])
    if bounding_boxes.__len__() == 6:
        true_values.append(img_src.replace(
            CAPTCHA_IMAGE_FOLDER, "").replace(".jpg", "").replace("_duplicate", ""))
        correctSeparations += 1
    return bounding_boxes


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)
# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
# captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)
final_predictions = []

# loop over the image paths
for image_file in captcha_image_files:
    # print(image_file)
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Add some extra padding around the image
    image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REPLICATE)

    letter_image_regions = erosionDilation(image, image_file)
    if len(letter_image_regions) != numChars:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    # loop over the letters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        try:
            letter_image = resize_to_fit(letter_image, 50, 50)
            pass
        except Exception:
            continue

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2),
                      (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))
    final_predictions.append(captcha_text)
    # Show the annotated image
    # cv2.imshow("Output", output)
    # cv2.waitKey()

print(true_values)
print(final_predictions)
print("We had ", correctSeparations, "correct separations")
print("We're predicting captchas with ", round(accuracy_score(
    true_values, final_predictions), 4)*100, "% accuracy")

true_letters, predicted_letters = flatten_string_lists(
    true_values, final_predictions)
print("We're predicting individual characters with ", round(accuracy_score(
    true_letters, predicted_letters), 4)*100, "% accuracy")
