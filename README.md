# Face Recognition Attendance System

This project is an automated Face Recognition–based Attendance System that identifies individuals from images or live video streams and records their attendance digitally. It replaces manual attendance with a computer vision and machine learning–based approach for higher accuracy and efficiency.

**System Workflow**

* Capture face images and create dataset
* Detect faces and extract embeddings
* Train SVM face recognition model
* Recognize faces in real time or uploaded images
* Mark and store attendance automatically

**Technical Details**
* Face Detection: SSD (ResNet-10) via OpenCV DNN
* Feature Extraction: OpenFace (128-D embeddings)
* Classifier: Support Vector Machine (Linear Kernel)
* Backend: Flask
* Data Storage: CSV files
