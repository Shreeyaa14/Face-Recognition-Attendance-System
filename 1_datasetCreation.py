import imutils
import time
import cv2
import csv
import os

# Load the cascade file
cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

# Get user input
Name = input("Enter your Name: ")
Roll_Number = input("Enter your Roll Number: ")

# Create a directory for the dataset if it doesn't exist
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

if not os.path.isdir(path):
    os.makedirs(path)
    print(f"Directory created for {sub_data}")

# Save user info in CSV
info = [Name, Roll_Number]
with open('student.csv', 'a', newline='') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)

print("Starting video stream...")
cam = cv2.VideoCapture(0)  # Use 0 if 1 doesn't work

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open video stream.")
    cam.release()
    cv2.destroyAllWindows()
else:
    time.sleep(2.0)
    total = 0

    while total < 50:
        print(f"Capturing image {total + 1} of 50")

        ret, frame = cam.read()

        # If frame not captured, exit
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize the frame and convert it to grayscale
        img = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Draw rectangles around detected faces and save images
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            p = os.path.join(path, f"{str(total).zfill(5)}.png")
            cv2.imwrite(p, img)
            total += 1

        # Show the video stream with rectangles around detected faces
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release camera and close all windows
    cam.release()
    cv2.destroyAllWindows()
