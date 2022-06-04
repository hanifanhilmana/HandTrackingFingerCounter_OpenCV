import cv2
import time
import os
import PyCharmModuleHandTracking as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0

detector = htm.handDetector(detectionCon=0.5)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    img = cv2.flip(img,1)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        print(fingers)
        totalFingers = fingers.count(1)
        #print(totalFingers)
        if fingers == [0, 0, 0, 0, 0]:
            total = 0
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [0, 1, 0, 0, 0]:
            total = 1
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [0, 1, 1, 0, 0]:
            total = 2
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [1, 1, 1, 0, 0]:
            total = 3
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [0, 1, 1, 1, 1]:
            total = 4
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [1, 1, 1, 1, 1]:
            total = 5
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [0, 1, 1, 1, 0]:
            total = 6
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [0, 1, 1, 0, 1]:
            total = 7
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [0, 1, 0, 1, 1]:
            total = 8
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (68, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [0, 0, 1, 1, 1]:
            total = 9
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (70, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

        if fingers == [1, 0, 0, 0, 0]:
            total = 10
            cv2.rectangle(img, (20, 15), (200, 200), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total), (8, 170), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)


        #cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        #cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    #10, (255, 0, 0), 25)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
