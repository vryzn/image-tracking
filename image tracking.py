import cv2
import numpy as np
from pyzbar.pyzbar import decode
import csv

# Load the image
image = cv2.imread('file.png')
output_image = image.copy()

# Function to detect green circles
def detect_green_circles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    blurred = cv2.GaussianBlur(mask, (9, 9), 2, 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    green_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            green_circles.append((center, radius))
            cv2.circle(output_image, center, radius, (0, 255, 0), 3)
    return green_circles

# Function to detect red X's
def detect_red_x(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    red_xs = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Assuming the lines form an 'X' where two intersect at an angle near 90 degrees
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                if np.abs(theta1 - theta2) > np.pi/4:
                    x = (np.cos(theta1)*rho1 + np.cos(theta2)*rho2) // 2
                    y = (np.sin(theta1)*rho1 + np.sin(theta2)*rho2) // 2
                    red_xs.append((int(x), int(y)))
                    cv2.circle(output_image, (int(x), int(y)), 10, (0, 0, 255), -1)
    return red_xs

# Function to detect barcodes
def detect_barcodes(image):
    barcodes = decode(image)
    barcode_info = []
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        barcode_info.append((barcode.data.decode("utf-8"), (x, y, w, h)))
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(output_image, barcode.data.decode("utf-8"), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return barcode_info

# Detect shapes and barcodes
green_circles = detect_green_circles(image)
red_xs = detect_red_x(image)
barcodes = detect_barcodes(image)

# Save the output image
cv2.imwrite('output_image.png', output_image)

# Generate the table output
with open('output_table.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Type', 'Location'])
    for circle in green_circles:
        writer.writerow(['Green Circle', circle[0]])
    for x in red_xs:
        writer.writerow(['Red X', x])
    for barcode in barcodes:
        writer.writerow(['Barcode', barcode[1]])

print("Processing complete. Outputs saved as 'output_image.png' and 'output_table.csv'.")
