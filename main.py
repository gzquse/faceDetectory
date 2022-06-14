import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('faces.jpg')
webcam = cv2.VideoCapture(0)

while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()
    # Must convert it to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # Draw rectangles around the faces [img, top left point, right bottom point, color, thickness]
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

    cv2.imshow('Clever Programmer Face qDetector', frame)
    key = cv2.waitKey(1)
    # Detect faces

    if key == 81 or key == 113:
        break

webcam.release()
