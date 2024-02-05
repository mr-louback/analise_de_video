import cv2 as cv

video_path = 'videos_1/video_8.mp4'
cap = cv.VideoCapture(video_path)
while cap.isOpened():
    new_width, new_height = 800, 600
    hog = cv.HOGDescriptor()
    cv.namedWindow('name window',cv.WINDOW_NORMAL)
    cv.resizeWindow('name window',new_width, new_height)
    cv.namedWindow('name window', cv.USAGE_DEFAULT)
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow('name window',frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break

if not cap.isOpened():
    print('in not opened!')
    exit()
cap.release()
cv.destroyAllWindows()
    

    


