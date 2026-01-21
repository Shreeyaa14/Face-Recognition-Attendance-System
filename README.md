# Face Recognition Attendance System

This project is an automated Face Recognition–based Attendance System that identifies individuals from images or live video streams and records their attendance digitally. It replaces manual attendance with a computer vision and machine learning–based approach for higher accuracy and efficiency.

**System Workflow**
* Capture face images and create dataset
* Detect faces and extract embeddings
* Train SVM face recognition model
* Recognize faces in real time or uploaded images
* Mark and store attendance automatically

**Technical Details**
* Programming Language: Python
* Face Detection: Deep Learning–based SSD with ResNet-10 backbone using OpenCV DNN
* Feature Extraction: OpenFace deep neural network generating 128-dimensional facial embeddings
* Face Recognition: Support Vector Machine (SVM) with linear kernel and probability estimation
* Real-Time Processing: Webcam-based video stream using OpenCV
* Web Framework: Flask
* Data Storage: CSV files
* Libraries Used: OpenCV, Scikit-learn, NumPy, Imutils, Pickle

**Output**
Dataset Creation Images and Input Images are in the Files.

<img width="686" height="429" alt="Screenshot 2026-01-21 103909" src="https://github.com/user-attachments/assets/23a47f76-b5a0-4483-9c7e-57311ec5493a" />

<img width="794" height="486" alt="Screenshot 2026-01-21 103859" src="https://github.com/user-attachments/assets/4f9099af-c48a-4df2-baf6-ef5a8dcdaecb" />


