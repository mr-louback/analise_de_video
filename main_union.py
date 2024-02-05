import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="This sample demonstrates the camshift algorithm."
)
parser.add_argument("image", type=str, help="path to image file")
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

inde_x = 0
while cap.isOpened():
    hog = cv.HOGDescriptor()
    ret, frame = cap.read()
    if not ret:
        break
    # take first frame of the video
    x, y, w, h = 400, 200, 100, 80
    track_window = (x, y, w, h)
    # set up the ROI for tracking
    roi = frame[y : y + h, x : x + w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(
        hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
    )
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 5, 1)
    # setup meanShift of window
    ret, track_window = cv.meanShift(dst, track_window, term_crit)
    # apply camshift to get the new location
    ret, track_window = cv.CamShift(dst, track_window, term_crit)
    x, y, w, h = track_window
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img1 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 1)
    img1 = cv.polylines(frame, [pts], 255, 1)
    cv.imshow("frame", img1)
    inde_x+=1
    k = cv.waitKey(30) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv.imwrite(f'mean_shift/mean_shift_{inde_x}.png',img1)

if not cap.isOpened():
    cv.imwrite(f'screenshots/mean_shift/mean_shift_{inde_x}.png', img1)
    print("not opened!")
    exit()

cap.release()
cv.destroyAllWindows()
