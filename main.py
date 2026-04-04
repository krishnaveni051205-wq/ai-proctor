from vision_module import VisionModule
from audio_module import AudioModule
import cv2
import numpy as np 
from logger_module import LoggerModule

vm = VisionModule()
am=AudioModule()
am.start_stream()
logger = LoggerModule()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Get the data from our new Module
    data = vm.process_frame(frame)
    if am.is_speech():
        cv2.putText(frame,"SPEECH DETECTED!",(20,430),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2) 
    
    # 2. DRAW OBJECTS (Phones, Books, etc.)
    for obj in data["objects"]:
        cv2.putText(frame, f"ALERT: {obj} detected!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 3. DRAW FACES & ALERTS
    for face in data["faces"]:
        x, y, w, h = face["box"]
        pitch, yaw, roll = face["pose"]
        
        # Draw the face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Logic for Head Pose Alerts
        if pitch < -10:
            cv2.putText(frame, "STATUS: LOOKING DOWN", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif abs(yaw) > 15:
            cv2.putText(frame, "STATUS: LOOKING SIDEWAYS", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: OK", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 4. Display the frame
    cv2.imshow("Master Proctor Dashboard", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    bar_width = int(am.current_volume * 1000)
cv2.rectangle(frame, (50, 400), (50 + bar_width, 420), (0, 255, 255), -1)
cv2.putText(frame, "Mic Level", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 3. Display the Warning Count
cv2.putText(frame, f"Audio Warnings: {am.alert_count}", (400, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

if am.alert_count > 5:
    cv2.putText(frame, "CRITICAL: PERSISTENT NOISE", (150, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
cap.release()
cv2.destroyAllWindows()