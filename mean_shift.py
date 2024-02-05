import numpy as np
import cv2 as cv
import argparse

# objects-
parser = argparse.ArgumentParser(
    description="This sample demonstrates the meanshift algorithm."
)
parser.add_argument("image", type=str, help="path to video")
args = parser.parse_args()
cap = cv.VideoCapture(args.image)
if not cap.isOpened():
    # take first frame of the video
    exit()
# video #6 com problema
# video #5 com problema
# video #2 com problema
# video #1 com problema

while cap.isOpened():
    ret, frame = cap.read()
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(hog.getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05,
    )
    if not ret:
        break
    inde_x = 0
    for x, y, w, h in boxes:
        track_window = (x, y, w, h)
        ret, frame = cap.read()
        roi = frame[y : h + y, x : w + x]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(
            hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
        )
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        dst = cv.calcBackProject([hsv_roi], [0], roi_hist, [0, 180], 1)
        term_crit = (cv.TERM_CRITERIA_COUNT, 10, 1)
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        image = cv.rectangle(frame, (x + w, y + h), (x, y), (0, 255, 0), 2)

        inde_x += 1
        cv.imshow("image", image)

    k = cv.waitKey(30) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("s"):
        cv.imwrite(f"mean_shift/mean_shift_{inde_x}.png", image)

cap.release()
cv.destroyAllWindows()
