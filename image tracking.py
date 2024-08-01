# brain dump
# from ashley: write script that imports a picture and identifies if there are green circles or red x's in the image
# should also output an image with shapes identified as well as a table outputting the number of shapes identified and physical locations within the image
# add capability to scan barcodes and note their locations
# code is based off code for license plate recognition

import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
import os
import tkinter as tk
from tkinter import filedialog


def sort_cont(character_contours):
    # to sort contours
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]

    (character_contours, boundingBoxes) = zip(
        *sorted(zip(character_contours, boundingBoxes), key=lambda b: b[1][i], reverse=False))

    return character_contours


def segment_chars(vapr_img, fixed_width):
    # extract value channel from HSV format of image and apply adaptive thresholding to reveal boxes with green circles or red x's

    V = cv2.split(cv2.cvtColor(vapr_img, cv2.COLOR_BGR2HSV))[2]

    thresh = cv2.adaptiveThreshold(V, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   11, 2)

    thresh = cv2.bitwise_not(thresh)

    # resize the region with the green circles and red x's to a canonical size
    vapr_img = imutils.resize(vapr_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # perform a connected components analysis and initialize the mask to store the locations of the character candidates
    labels = measure.label(thresh, background=0)

    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    # loop over the unique components
    characters = []
    for label in np.unique(labels):

        # if this is the background label ignore it
        if label == 0:
            continue
        # otherwise construct the label mask to display
        # only connected components for the current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:

            # grab the largest contour which corresponds to the component in the mask, then grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solidity, and height ratio for the componenet
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(vapr_img.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass the rules tests
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                # compute the convex hull of the contour and draw it on the character candidates
                # mask
                hull = cv2.convexHull(c)

                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        contours, hier = cv2.findContours(charCandidates,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sort_cont(contours)

    # value to be added to each dimension of the character
    addPixel = 4
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if y > addPixel:
            y = y - addPixel
        else:
            y = 0
        if x > addPixel:
            x = x - addPixel
        else:
            x = 0

            temp = bgr_thresh[y:y + h + (addPixel * 2),
                   x:x + w + (addPixel * 2)]

            characters.append(temp)

            return characters



    else:
        return None


class CircleFinder:
    def __init__(self, minCircleArea, maxCircleArea):

        # minimum area of the green circles
        self.min_area = minCircleArea

        # maximum area of the green circles
        self.max_area = maxCircleArea

        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 5))

    def preprocess(self, input_img):

        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)

        # convert to gray
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

        # sobelx to get the vertical edges
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

        # otsu's thresholding
        ret2, threshold_img = cv2.threshold(sobelx, 0, 225,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = self.element_structure
        morph_n_threshold_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img,
                         op=cv2.MORPH_CLOSE,
                         kernel=element,
                         dst=morph_n_threshold_img)

        return morph_n_threshold_img

    def extract_contours(self, after_preprocess):

        contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_circle(self, circle):

        gray = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]

            # inddex of the largest contour in the area
            # array
            max_index = np.argmax(areas)

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            if not self.ratioCheck(max_cntArea, circle.shape[1], circle.shape[1]):
                return circle, False, None

            return circle, True, [x, y, w, h]

        else:
            return circle, False, None

    def check_circle(self, input_img, contour):

        min_rect = cv2.minAreaRect(contour)

        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_circle_img, circleFound, coordinates = self.clean_circle(after_validation_img)

            if circleFound:
                circles_on_boxes = self.find_circles_on_boxes(after_clean_circle_img)

                if (circles_on_boxes is not None and len(circles_on_boxes) == any):
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    after_check_circle_img = after_clean_circle_img

                    return after_check_circle_img, circles_on_boxes, coordinates

                return None, None, None

    def find_possible_circles(self, input_img):

        # finding all possible contours that can be circles

        circles = []
        self.col_on_circle = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_circle_contours = self.extract_contours(self.after_preprocess)

        for cnts in possible_circle_contours:
            circles, circles_on_boxes, coordinates = self.check_circle(input_img, cnts)

            if circles is not None:
                circles.append(circles)
                self.circles_on_boxes.append(circles_on_boxes)
                self.corresponding_area.append(coordinates)

            if (len(circles) > 0):
                return circles

            else:
                return None

        def find_circles_on_boxes(self, circles):

            circlesFound = segment_chars(circles, 400)
            if circlesFound:
                return circlesFound

        # Green circles on boxes features
        def ratioCheck(self, area, width, height):

            min = self.min_area
            max = self.max_area

            ratioMin = 10
            ratioMax = 50

            ratio = float(width) / float(height)

            if ratio < 1:
                ratio = 1 / ratio

                if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
                    return False

                return True

        def preRatioCheck(self, area, width, height):

            min = self.min_area
            max = self.max_area

            ratioMin = 7
            ratioMax = 55

            ratio = float(width) / float(height)

            if ratio < 1:
                ratio = 1 / ratio

            if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
                return False

            return True

        def validateRatio(self, rect):
            (x, y), (width, height), rect_angle = rect

            if (width > height):
                angle = -rect_angle
            else:
                angle = 90 + rect_angle

            if angle > 15:
                return False

            if (height == 0 or width == 0):
                return False

            area = width * height

            if not self.preRatioCheck(area, width, height):
                return False
            else:
                return True


class OCR:

    def __init__(self, modelFile, labelFile):

        self.model_file = modelFile
        self.label_file = labelFile
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto())

    def load_graph(self, modelFile):

        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()

        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())

            with graph.as_default():
                tf.import_graph_def(graph_def)

                return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()

        for l in proto_as_ascii_lines:
            label.append(l.rstrip())

            return label

    def convert_tensor(self, image, imageSizeOuput):

        # takes an image and transforms it in tensor

        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)

        np_image_data = np.array(image)
        np_image_data = cv2.normalize(np_image_data.astyoe('float'), None, -0.5, .5, cv2.NORM_MINMAX)

        np_final = np.expend_dims(np_image_data, axis=0)

        return np_final

    def label_image(self, tensor):

        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]

        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        circles = ""

        for img in listImages:

            if cv2.waitkey(25) & 0xFF == ord('q'):
                break
            circles = circles + self.label_image(self.convert_tensor(img, imageSizeOuput))

            return circles, len(circles)


# main function to perform the whole task in a sequence
if __name__ == "__main__":

    findCircles = CircleFinder(minCircleArea=4100, maxCircleArea=15000)
    model = OCR(modelFile="model/binary_128_0.50_ver3.pb", labelFile="model/binary_128_0.50_labels_ver2.txt")

    # use GUI to open the file
    root = tk.TK()
    root.withdraw()  # hide root window
    file_path_start = 'C/Users/vroyzen/OneDrive - Product Insight/Documents'
    file_path = filedialog.askopenfilename(filetypes=[("PNG Files", ".png")], initialfile=file_path_start)

    # checking to confirm that user selected a file or canceled operation
    if not file_path_start:
        print("path does not exist. Check file path directory and make sure there is a match")
        exit()
    if not file_path:
        print("No file selected. Exiting...")
        exit()

    while (file_path.isOpened()):
        ret, img = file_path.read()

        if ret == True:
            cv2.imshow('original photo', img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            possible_circles = findCircles.find_possible_circles(img)
            if possible_circles is not None:

                for i, p in enumerate(possible_circles):
                    circles_on_boxes = findCircles.circles_on_boxes[i]
                    recognized_circles, _ = model.label_image_list(circles_on_boxes, imageSizeOuput=128)

                    print(recognized_circles)
                    cv2.imshow('circles', p)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
        else:
            break

    file_path.release()
    cv2.destroyAllWindows()