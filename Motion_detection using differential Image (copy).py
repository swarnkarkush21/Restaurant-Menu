import cv2
import numpy as np
import time
count = 0
bg = cv2.bgsegm.createBackgroundSubtractorMOG()
def DiffFrame(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return  cv2.bitwise_and(d1, d2)

def closing(mask):
    kernel = np.ones((7,7),np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closing

def dilsion(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilate = cv2.dilate(mask, kernel, iterations = 2)
    return dilate

def erosion(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    erode = cv2.erode(mask, kernel, iterations = 2)
    return erode

def opening(mask):
    kernel = np.ones((6,6),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opening

# def select_point(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
#         roiPts.append((x,y))
#         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#         cv2.imshow("frame", frame)

cap = cv2.VideoCapture(0)
winName = "Movement Detector"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
ts = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
t =  cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
ta =  cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while True:
    ret, frame = cap.read()
    ret, thresh = cv2.threshold(DiffFrame(ts, t, ta), 50, 255, cv2.THRESH_BINARY)
    closing(thresh)
    opening(thresh)
    roi = DiffFrame(ts, t, ta)
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
    roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((10., 70., 40.)), np.array((40., 170., 90.)))
    mean = cv2.mean(frame, mask=mask)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    dilsion(mask)
    erosion(mask)
    w, h = thresh.shape[:2]
    new_frame = bg.apply(thresh)
    skin = cv2.bitwise_and(new_frame, new_frame, mask=mask)
    i, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[:10]
    cv2.drawContours(frame, cnts, -1, (0, 255, 0), 1)
    #pixelpoints = np.transpose(np.nonzero(mask))
    # pixelpoints = cv2.findNonZero(mask)
    for i, c in enumerate(cnts[-1:]):
        #c = max(cnts, cv2.contourArea(np.int0(c[i])))
        # contours property
        area = cv2.contourArea(c)
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(new_frame, mask=mask)
        leftmost = tuple(c[c[:, :, 0].argmin()][0])
        rightmost = tuple(c[c[:, :, 0].argmax()][0])
        topmost = tuple(c[c[:, :, 1].argmin()][0])
        bottommost = tuple(c[c[:, :, 1].argmax()][0])
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        print 'cnt point', c,'area', area, 'aspect_ratio', aspect_ratio, 'extent', extent, 'solidity', solidity
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        track_window = x, y , w, h
        ret, track_window = cv2.CamShift(thresh, track_window , term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.putText(frame, 'Tracked', (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, 'Tracked', (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        t = time.time()
        localtime = time.strftime("%a, %d %b %Y %H:%M:%S +05:30", time.localtime(t))
        cv2.putText(frame, str(localtime), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.rectangle(frame, (x, y), ((x+w) / 2, (y+h) / 2), (0, 255, 255), 2)
        cv2.circle(frame, ((x + w) / 2, (y + w) / 2), 200, (255, 255, 0), 2)
        count = count + 1
        cv2.putText(frame, str(count), (x + 100, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Threshold',thresh)
    cv2.imshow('mask', skin)
    cv2.imshow(winName, DiffFrame(ts, t, ta))
    cv2.imwrite("/home/kush/Desktop/hhh.jpg", frame)
    cv2.imshow('frame', frame)
    #cv2.imshow('backproj', backproj)
    ts = t
    t = ta
    ta = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
print "Count", count
cap.release()
cv2.destroyAllWindows()