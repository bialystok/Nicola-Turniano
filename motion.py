import numpy as np
import cv2
import imutils
import random
from imutils.object_detection import non_max_suppression

def sameRect(rect1, rect2):
    # if(rect1 == None or rect2 == None):
    #    return False
    # return False
    if (rect1[0] > rect2[2] or rect2[0] > rect1[2]):
        print("if 1: return false")
        return False
    elif (rect1[3] < rect2[1] or rect2[3] < rect1[1]):
        print("if 2: return false")
        return False
    else:
        print("return true")
        return True
    pass

def samequad(x1, y1, x2, y2, vett):
    #return 0
    for i in range(len(vett)):
        P_A_overlap = ((max(x1, vett[i]) - min(x2, vett[i + 1])) * (max(y1, vett[i + 2]) - min(y2, vett[i + 3]))) / (
                    (x2 - x1) * (y2 - y1))
        print(P_A_overlap)
        i = i + 3
        if (P_A_overlap > 50):
            return i - 3;
    return -1;


minArea = 400
display_padding = 5
fuse_padding = 0
fuse_padding_bottom = 0
sq = []
colors = []
#color0 = (0, 255, 0)

# cap = cv2.VideoCapture(r'/Users/nicola/Desktop/video/0-40_1-05__3_13.mp4')
cap = cv2.VideoCapture(r'/Users/nicola/Desktop/video/192.168.101.19_ch29_20190704053054_20190704080049.mp4')
# cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
# kernel = np.ones((3,3),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
while 1:
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=750)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.blur(gray, (5, 5))
    # gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # gray = cv2.medianBlur(gray, 21)
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, kernel2, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1, iterations=3)
    # fgmask = cv2.dilate(fgmask, kernel1, iterations=4)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel3, iterations=4)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel1, iterations=3)
    cv2.imshow('mask', fgmask)
    # thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = fgmask
    # thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print(cnts)
    # ennepi=np.array([[x, y, x + w, y + h] for (x, y, w, h) in cnts])
    # pick = non_max_suppression(ennepi, probs=None, overlapThresh=0.65)
    # loop over the contours
    rects = []
    for c in cnts:
        if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] >= minArea:
            rects.append(cv2.boundingRect(c))
    # print("rects", rects)
    rects = np.array(
        [[x - fuse_padding, y - fuse_padding, x + w + fuse_padding, y + h + fuse_padding + fuse_padding_bottom] for
         (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)
    # print("pick", pick)
    if len(pick) > 0:
        pick = pick.tolist()
    new_squares = []
    new_colors = []
    for i in range(len(pick)):
        found = False
        for j in range(len(sq)):

            if sameRect(pick[i], sq[j]):
                new_squares.append(pick[i])
                new_colors.append(colors[j])
                color0 = colors[j]
                found = True
                break
        if not found:
            new_squares.append(pick[i])
            color1 = random.randint(0, 254)
            color2 = random.randint(0, 254)
            color3 = random.randint(0, 254)
            color0 = (color1, color2, color3)
            new_colors.append(color0)
    for i in range (len(new_squares)):
        cv2.rectangle(frame, (new_squares[i][0] + fuse_padding - display_padding, new_squares[i][1] + fuse_padding - display_padding),
                      (new_squares[i][2] - fuse_padding + display_padding, new_squares[i][3] - fuse_padding - fuse_padding_bottom + display_padding),
                      new_colors[i], 2)

    sq = new_squares
    colors = new_colors
    k = cv2.waitKey(30) & 0xff
    cv2.imshow("frame", frame)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
