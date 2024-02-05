import cv2 as cv
import os 

capture = cv.VideoCapture('videos_0/video_2.mp4')
if not capture.isOpened():
    print('not opened')
display_interval = 3
frame_count = 0

while capture.isOpened():
    width, heigth = 700, 500
    cv.namedWindow('people', cv.WINDOW_NORMAL)
    cv.resizeWindow('people', width, heigth)
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(hog.getDefaultPeopleDetector())
    hog.setSVMDetector(hog.getDaimlerPeopleDetector())

    ret, frame = capture.read()
    if not ret:
        break
    if frame_count % display_interval == 0:
        boxes, weights = hog.detectMultiScale(frame.copy(), winStride=(8, 8), padding=(4,4), scale=1.06)
        for (x, y, w, h) in boxes:#back
            file_folder = 'images/people_hog.getDaimlerPeopleDetector'
            file_path = os.path.join(os.getcwd(), file_folder)
            area = cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0),2)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                is_not_none_file = os.path.join(file_path, f'people_{frame_count}.jpg')
                cv.imwrite(is_not_none_file, area)
                print(f'A people_{frame_count}.jpg foi criada.')
                cv.imshow('people', area)#front

            if area is not None:
                is_not_none_file = os.path.join(file_path, f'people_{frame_count}.jpg')
                cv.imwrite(is_not_none_file, area)
                print(f'A imagem foi salva em {is_not_none_file}.')
                cv.imshow('people', area)#front

            else: 
                print('Erro ao ler a imagem.')
                # cv.imshow('car_vision', area)#front
    frame_count += 1
    key = cv.waitKey(1)
    if key == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
