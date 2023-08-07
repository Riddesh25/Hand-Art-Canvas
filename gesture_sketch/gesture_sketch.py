import cv2
import mediapipe as mp
import numpy as np
import os
import math
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# captures video from the default camera (index 0) and makes some configurations to the capture settings:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 5)
width = 1280
height = 720
cap.set(3, width)
cap.set(4, height)

# creates an empty canvas (image) using NumPy, with the dimensions defined by the height and width variables, data type set to unsigneg 8-bit integer
imgCanvas = np.zeros((height, width, 3), np.uint8)

# list all the files and directories present in the specified folderPath
folderPath = 'Colorpanel'
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# Presettings:
header = overlayList[0]
drawColor = (0, 0, 255)
thickness = 20 # Thickness of the painting
tipIds = [4, 8, 12, 16, 20] # Fingertips indexes
xp, yp = [0, 0] # Coordinates that will keep track of the last position of the index finger

# mediapipe library's hand tracking module (mp_hands) to detect and track hands in a video stream
with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # pre-processing steps include flipping the image horizontally and converting it from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # image is marked as non-writeable to improve performance
        image.flags.writeable = False
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Getting all hand points coordinates
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                # condition checks if any hand landmarks were detected and recorded in the points list
                if len(points) != 0:
                    x1, y1 = points[8]  # Index finger
                    x2, y2 = points[12] # Middle finger
                    x3, y3 = points[4]  # Thumb
                    x4, y4 = points[20] # Pinky

                    ## Checking which fingers are up
                    fingers = []
                    # checks whether the thumb is up
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Checks the remaining fingers 
                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    ## Selection Mode - Which two fingers need to be up
                    nonSel = [0, 3, 4] # indexes of the fingers that need to be down in the Selection Mode
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                        xp, yp = [x1, y1]

                        # Selecting the different colors and the eraser on the screen
                        if(y1 < 125):
                            if(120 < x1 < 220):
                                header = overlayList[0]
                                drawColor = (0, 0, 255)
                            elif(290 < x1 < 410):
                                header = overlayList[1]
                                drawColor = (0, 255, 0)
                            elif(470 < x1 < 580):
                                header = overlayList[2]
                                drawColor = (255, 0, 0)
                            elif(660 < x1 < 780):
                                header = overlayList[3]
                                drawColor = (0, 255, 255)
                            elif(850 < x1 < 960):
                                header = overlayList[4]
                                drawColor = (128, 0, 128)
                            elif(1120 < x1 < 1240):
                                header = overlayList[5]
                                drawColor = (0, 0, 0)

                        cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                    ## Stand by Mode - Checks when the index and the pinky fingers are open and stops drawing
                    nonStand = [0, 2, 3] # indexes of the fingers that need to be down in the Stand Mode
                    if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in nonStand):
                        # The line between the index and the pinky indicates the Stand by Mode
                        cv2.line(image, (xp, yp), (x4, y4), drawColor, 5) 
                        xp, yp = [x1, y1]

                    ## Draw Mode - to draw with the index finger
                    nonDraw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                        # The circle in the index finger indicates the Draw Mode
                        cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED) 
                        if xp==0 and yp==0:
                            xp, yp = [x1, y1]
                        # Draws a line between the current position and the last position of the index finger
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        # Update the previous position
                        xp, yp = [x1, y1]

                    ## Clear the canvas when the hand is closed
                    if all(fingers[i] == 0 for i in range(0, 5)):
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = [x1, y1]

                    ## Adjust the thickness of the line using the index finger and thumb
                    selecting = [1, 1, 0, 0, 0] # to Select the thickness of the line
                    setting = [1, 1, 0, 0, 1]   # for Setting the thickness
                    if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(fingers[i] == j for i, j in zip(range(0, 5), setting)):

                        # Obtaining the circle's radius, which represents the thickness of the drawing, 
                        #by measuring the distance between the index finger and the thumb.
                        r = int(math.sqrt((x1-x3)**2 + (y1-y3)**2)/3)
                        
                        # Getting the mid-point between these two fingers
                        x0, y0 = [(x1+x3)/2, (y1+y3)/2]
                        
                        # Getting the vector that is orthogonal to the line formed between two fingers
                        v1, v2 = [x1 - x3, y1 - y3]
                        v1, v2 = [-v2, v1]

                        # Normalization
                        mod_v = math.sqrt(v1**2 + v2**2)
                        v1, v2 = [v1/mod_v, v2/mod_v]
                        
                        # Draw the circle that represents the draw thickness in (x0, y0) and orthogonaly 
                        # translated c units
                        c = 3 + r
                        x0, y0 = [int(x0 - v1*c), int(y0 - v2*c)]
                        cv2.circle(image, (x0, y0), int(r/2), drawColor, -1)

                        # Final Setting of the thickness chosen as the pinky finger is pointed up
                        if fingers[4]:                        
                            thickness = r
                            cv2.putText(image, 'Check', (x4-25, y4-8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 1)

                        xp, yp = [x1, y1]

        # Setting the header in the video
        image[0:125, 0:width] = header

        # The image processing will generate the camera image with the drawing made on imgCanvas
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.imshow('MediaPipe Hands', img)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()