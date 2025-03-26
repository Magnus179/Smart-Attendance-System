import imutils
import time
import cv2
import csv
import os
import pandas as pd


cascade_path = "C:/Users/nanim/Downloads/attendence Recognition-20240714T145610Z-001/attendence Recognition/code/haarcascade_frontalface_default.xml"

# Check if the cascade classifier file exists
if not os.path.isfile(cascade_path):
    raise RuntimeError(f"File {cascade_path} not found")

detector = cv2.CascadeClassifier(cascade_path)

# Check if the cascade classifier loaded successfully
if detector.empty():
    raise RuntimeError("Error: Failed to load cascade classifier.")

Name = str(input("Enter your Name : "))
Roll_Number = int(input("Enter your Roll_Number : "))
dataset = 'C:/Users/nanim/Downloads/attendence Recognition-20240714T145610Z-001/attendence Recognition/dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)
    print(sub_data)

info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)
csvFile.close()

print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)
total = 0

while total < 50:
    print(total)
    _, frame = cam.read()
    img = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.sep.join([path, "{}.png".format(
            str(total).zfill(5))])
        cv2.imwrite(p, img)
        total += 1

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
