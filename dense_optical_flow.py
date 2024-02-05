#Dense Optical Flow
import numpy as np
import cv2 as cv
import argparse


parser = argparse.ArgumentParser(description="This sample demonstrates the camshift algorithm.")
parser.add_argument("image", type=str, help="path to image file")
args = parser.parse_args()
cap = cv.VideoCapture(args.image)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
inde_x = 0
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY) # COLOR_BGR2GRAY
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv.imwrite(f'screenshots/fboptical/opticalfb_{inde_x}.png', frame2)
        cv.imwrite(f'screenshots/hsvoptical/opticalhsv_{inde_x}.png', bgr)
    prvs = next
    inde_x += 1
cv.destroyAllWindows()