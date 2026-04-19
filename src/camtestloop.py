import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    print(i, ret)
    cap.release()