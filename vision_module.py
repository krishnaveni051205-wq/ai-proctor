import cv2
import numpy as np
from ultralytics import YOLO

class VisionModule:
    def __init__(self, lbf_model_path="lbfmodel.yaml", yolo_model_path="yolov8n.pt"):
        # 1. Load Face & Landmark Models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(lbf_model_path)
        
        # 2. Load Object Detection (YOLO)
        self.yolo_model = YOLO(yolo_model_path)
        self.unauthorized_objects = ['cell phone', 'book', 'laptop']

        # 3. 3D Model Points for Head Pose
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def get_head_pose(self, marks, frame_shape):
        image_points = np.array([marks[30], marks[8], marks[36], marks[45], marks[48], marks[54]], dtype="double")
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        
        _, rot_vec, trans_vec = cv2.solvePnP(self.model_points, image_points, camera_matrix, np.zeros((4,1)))
        rmat, _ = cv2.Rodrigues(rot_vec)
        # Using the fix from Day 7 for the unpacking error
        res = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))
        angles = res[0].flatten()[:3]
        return angles # [pitch, yaw, roll]

    def process_frame(self, frame):
        results = {"faces": [], "objects": [], "alerts": []}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Object Detection
        yolo_results = self.yolo_model(frame, stream=True, verbose=False, conf=0.5)
        for r in yolo_results:
            for box in r.boxes:
                cls_name = self.yolo_model.names[int(box.cls[0])]
                if cls_name in self.unauthorized_objects:
                    results["objects"].append(cls_name)

        # Face & Pose Detection
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            _, landmarks = self.facemark.fit(gray, np.array([(x, y, w, h)]))
            for marks in landmarks:
                pitch, yaw, roll = self.get_head_pose(marks[0], frame.shape)
                results["faces"].append({"box": (x,y,w,h), "pose": (pitch, yaw, roll)})
        
        return results