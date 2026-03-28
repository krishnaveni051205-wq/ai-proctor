import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
facemark=cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")
def get_gaze_ratio(eye_points,facial_landmarks,frame,gray):
    mask=np.zeros(frame.shape[:2],dtype=np.uint8)
    region=np.array([(facial_landmarks[i][0], facial_landmarks[i][1]) for i in eye_points],np.int32)
    cv2.fillPoly(mask,[region],255)
    eye=cv2.bitwise_and(gray,gray,mask=mask)
    margin=2
    min_x=np.min(region[:,0])-margin
    max_x=np.max(region[:,0])+margin
    min_y=np.min(region[:,1])-margin
    max_y=np.max(region[:,1])+margin
    threshold_eye=eye[min_y:max_y,min_x:max_x]
    _, threshold_eye=cv2.threshold(threshold_eye,70,255,cv2.THRESH_BINARY_INV)
    height,width=threshold_eye.shape
    left_side = threshold_eye[0:height, 0:int(width/2)]
    left_white = cv2.countNonZero(left_side)
    right_side = threshold_eye[0:height, int(width/2):width]
    right_white = cv2.countNonZero(right_side)
    if left_white == 0: return 1 # Avoid division by zero
    return right_white / left_white
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        _, landmarks = facemark.fit(gray, np.array([(x, y, w, h)]))
        for marks in landmarks:
            # Indices for eyes in 68-point model: Left(36-41), Right(42-47)
            ratio_l = get_gaze_ratio(range(36, 42), marks[0], frame, gray)
            ratio_r = get_gaze_ratio(range(42, 48), marks[0], frame, gray)
            gaze_ratio = (ratio_l + ratio_r) / 2
            if gaze_ratio < 0.8:
                text = "LOOKING RIGHT"
            elif gaze_ratio > 2.0:
                text = "LOOKING LEFT"
            else:
                text = "CENTER"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("AI Proctor - Day 6 Gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()