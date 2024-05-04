import cv2
import numpy as np
import solver
from utils import *


pathImage = "Resources/3.png"
heightImg = 450
widthImg = 450

model = initializePredictionModel()

#Image Preparation

img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((widthImg, heightImg, 3), np.uint8)
imgThreshold = preProcess(img)


#Contour

imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0,255,0), 3)

#Biggest Contour

biggest, maxArea = biggestContour(contours)
#print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0,0,255), 20) #draw main sudoku box
    pts1 = np.float32(biggest) #for warp
    pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) #final points for warp
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

#Split image and identify digits

imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpColored)
print(len(boxes))
#cv2.imshow("sample", boxes[1])
numbers = getPrediction(boxes, model)

print(numbers)
imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255,0,255))
numbers = np.array(numbers)
posArray = np.where(numbers > 0, 0, 1)

#Getting solution
board = np.array_split(numbers, 9)

try:
    solver.solve(board)
except:
    pass

flatlist = []
for sublist in board:
    for item in sublist:
        flatlist.append(item)
solvedNumbers = flatlist*posArray
imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

#Final display
pts2 = np.float32(biggest)
pts1 = np.float32([[0,0], [widthImg,0], [0, heightImg], [widthImg, heightImg]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgInvWarpColored = img.copy()
imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
imgDetectedDigits = drawGrid(imgDetectedDigits)
imgSolvedDigits = drawGrid(imgSolvedDigits)

imageArray = ([img, imgThreshold, imgContours, imgWarpColored],
              [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
stackedImage = stackImages(imageArray, 1)
cv2.imshow('Stacked Images', stackedImage)
#cv2.destroyAllWindows()

cv2.waitKey(0)