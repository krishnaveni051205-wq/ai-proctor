import cv2
import numpy as np
# 1. Load the Face Detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Create the Facemark LBF object and load the model
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find faces first
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Create a rectangle for the face found
        face_rect = (x, y, w, h)
        
        # 3. Detect landmarks within that face rectangle
        # Note: Facemark expects a list of faces, so we wrap face_rect in [ ]
        err, landmarks = facemark.fit(gray, np.array([face_rect]))

        if err:
            for marks in landmarks:
                for point in marks[0]:
                    px, py = int(point[0]), int(point[1])
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

    cv2.imshow("AI Proctor - Day 5 OpenCV LBF", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()