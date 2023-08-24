import cv2
import numpy as np
import utils

# webcam or from file
'''
webcam_feed = False
cap = cv2.VideoCapture(1)
cap.set(10, 160)
'''


questions, choices = 5, 5

img_height, img_width = 700, 700
ans_key = [0, 1, 2, 0, 3]

student_score = {}

path = 'Marked3.jpg'

img = cv2.imread(path)
img = cv2.resize(img, (img_height, img_width))  # to fit the window

img_blank = np.zeros((img_height, img_width, 3), np.uint8)  # for testing purposes

# image pre-processing
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to gray
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)  # adding gaussian blur
img_canny = cv2.Canny(img_blur, 10, 70)  # apply canny

# detecting the lines in the given image
img_contour = img.copy()  # so that our original image in not affected
img_grade = img.copy()  # so the original image is unaffected
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# highlighting the lines
cv2.drawContours(img_contour, contours, -1, (255, 0, 0), 5)

# in our image our concerned area is the biggest rectangle inside which bubbles are there
rectangle_contours = utils.rectContour(contours)  # filtering rectangle contours

biggest_rect = utils.getCornerPoints(rectangle_contours[0])  # gets 4 points which might not be in order

# spotting the required rectangle
biggest_rect_points = utils.reorder(biggest_rect)  # arranging in order
cv2.drawContours(img_grade, biggest_rect_points, -1, (0, 0, 255), 15)

# cropping our required rectangle
pts1 = np.float32(biggest_rect_points)
pts2 = np.float32(
    [[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])  # which point corresponds to which point

matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_warp = cv2.warpPerspective(img, matrix, (img_width, img_height))

# Threshold, converting the image pixel to it's opposite so as to make detection easier
img_wrap_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY);  # converting to grayscale cause can't do that with RGB
img_threshold = cv2.threshold(img_wrap_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]  # Apply threshold to every pixel

# overlaying the image with grid and each cell will have our image
grid_cells = utils.splitBoxes(img_threshold)

count_rows, count_cols = 0, 0

# creating a storage matrix to store the non zero value
pix_sto_mat = np.zeros((questions, choices))

for i in grid_cells:
    total_pixels = cv2.countNonZero(i)
    pix_sto_mat[count_rows][count_cols] = total_pixels
    count_cols += 1

    if count_cols == choices:
        count_cols = 0
        count_rows += 1

# finally find the answer and put them in our list
# cell with max pixel is our answer
marked_answers = []
for i in range(0, questions):
    temp = pix_sto_mat[i]

    max_index = np.where(temp == np.amax(temp))
    marked_answers.append(max_index[0][0])

# now we are comparing with answer key for final score

score = []
for i in range(0, questions):
    if marked_answers[i] == ans_key[i]:
        score.append(10)
    else:
        score.append(0)

final_score = sum(score)


# to store roll number and score
# detecting the roll number
roll_num_contour = utils.getCornerPoints(rectangle_contours[2])
cv2.drawContours(img_contour, roll_num_contour, -1, (255, 0, 0), 5)
roll_num_contour = utils.reorder(roll_num_contour)
ptr1 = np.float32(roll_num_contour)
ptr2 = np.float32([[0, 0], [900, 0], [0, 100], [900, 100]])
matrix2 = cv2.getPerspectiveTransform(ptr1, ptr2)
roll_img_warp = cv2.warpPerspective(img, matrix2, (400, 400))

roll_num = utils.get_roll_num(roll_img_warp)

student_score[roll_num] = final_score

for roll in student_score:
    print("Roll Num " + str(roll) + " Has Scored " + str(student_score[roll]))

'''

# showing final score on actual image
grade_contour = utils.getCornerPoints(rectangle_contours[1])
grade_contour = utils.reorder(grade_contour)
cv2.drawContours(img, grade_contour, -1, (255, 0, 0),5)
ptrG = np.float32(grade_contour)
ptrG2 = np.float32([[0,0], [325, 0], [0, 150], [325, 150]])

matrxG = cv2.getPerspectiveTransform(ptrG, ptrG2)
grade_img_warp = cv2.warpPerspective(img, matrxG, (325, 150))

final_img = img
#cv2.imshow("Original", img)

'''

img_arr = ([img, img_gray, img_blur, img_canny],
           [img_contour, img_grade, img_warp, img_threshold])

labels = [['Original', 'Gray Image', 'Blur Image', 'Canny Image'],
          ['Contours Image', 'Rect Points', 'Warped Image', 'Thresh Image']]

stacked_images = utils.stackImages(img_arr, 0.45, labels)

#cv2.imshow('Flow', stacked_images)
cv2.imshow('Marked1', img)

cv2.waitKey(0)
