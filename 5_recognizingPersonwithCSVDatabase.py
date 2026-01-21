from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import cv2
import os
import csv
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)

# Paths to the model and data files
embeddingModel = "openface_nn4.small2.v1.t7"
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
conf = 0.5
unknown_name = "Unknown"  # Label for untrained faces

# Load the face detector model
print("Loading face detector...")
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load the face embedding model
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Load the SVM recognizer and label encoder
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Flask route for uploading and processing the image
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join("uploads", filename)
            file.save(filepath)

            # Call function to process the image and record attendance
            attendance_report = process_image(filepath)
            return render_template("attendance_report.html", attendance=attendance_report)

    return render_template("upload.html")

def process_image(image_path):
    # Load the group photo
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load image.")
        return "Error loading image."

    # Prepare to record attendance
    attendance = {}
    absentees = set()

    # Resize the image for faster processing and get dimensions
    frame = cv2.resize(frame, (600, int(600 * frame.shape[0] / frame.shape[1])))
    (h, w) = frame.shape[:2]

    # Detect faces in the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Loop through detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Skip small detections
            if fW < 20 or fH < 20:
                continue

            # Create an embedding for the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j] if proba > conf else unknown_name  # If the confidence is low, mark as Unknown

            # Mark attendance
            if name != unknown_name:
                attendance[name] = "Present"
            else:
                absentees.add(name)

            # Draw a bounding box and label on the image
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Save attendance to a CSV file with UTF-8 encoding
    with open("attendance.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header if the file is empty
        if os.stat("attendance.csv").st_size == 0:
            writer.writerow(["Name", "Status"])

        # Write attendance data
        for name, status in attendance.items():
            writer.writerow([name, status])
        
        # Write absent persons as well
        for name in absentees:
            writer.writerow([name, "Absent"])

    print("Attendance recorded for:", ", ".join(attendance.keys()))

    return attendance

if __name__ == "__main__":
    app.run(debug=True)
