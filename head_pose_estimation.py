import cv2
import numpy as np

# Load models from previous days
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

cap = cv2.VideoCapture(0)

# 3. Standard 3D model points of a human face
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        _, landmarks = facemark.fit(gray, np.array([(x, y, w, h)]))
        for marks in landmarks:
            marks = marks[0]
            # Extract the 6 specific 2D points
            image_points = np.array([
                marks[30], marks[8], marks[36], marks[45], marks[48], marks[54]
            ], dtype="double")

            # Camera internals
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4,1)) 

            # Solve PnP
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            # Convert rotation vector to angles
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _, *extra = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
            pitch, yaw, roll = angles.flatten()[:3]

            # Logic for "Searching for Notes" (Looking Down)
            status = "Status: OK"
            color = (0, 255, 0)
            if pitch < -10: # Threshold for looking down
                status = "ALERT: LOOKING DOWN!"
                color = (0, 0, 255)
            elif abs(yaw) > 15: # Threshold for looking sideways
                status = "ALERT: LOOKING SIDEWAYS!"
                color = (0, 0, 255)

            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("AI Proctor - Day 7 Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()