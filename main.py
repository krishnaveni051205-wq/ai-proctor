from vision_module import VisionModule
from audio_module import AudioModule
import cv2
import numpy as np 
from logger_module import LoggerModule
import os
import time
if not os.path.exists("evidence"):
    os.makedirs("evidence")
vm = VisionModule()
am=AudioModule()
am.start_stream()
logger = LoggerModule()
cap = cv2.VideoCapture(0)
is_recording=False
record_timer=0
video_out=None
last_alert_count=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Get the data from our new Module
    data = vm.process_frame(frame)
    if am.is_speech():
        transcript=am.get_transcript()
        if transcript and transcript != "[Unclear Speech]":
            cv2.putText(frame,f"HEARD:{transcript}",(20,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            logger.log_event("SPEECH_TRANSCRIPT", transcript)
    liveness_status="LIVE" if am.is_live else "RECORDING DETECTED"
    liveness_color=(0,255,0) if am.is_live else (0,0,255)
    cv2.putText(frame,f"Source:{liveness_status}",(20,100),
    cv2.FONT_HERSHEY_SIMPLEX,0.5,liveness_color,2)
    #mouth aspect ratio to check if the user is speaking or someone behind him is
    is_mouth_open=data.get("mouth_open",False)
    is_speech_active=am.is_speech()
    if is_speech_active and not is_mouth_open:
        cv2.putText(frame,"ALERT: EXTERNAL VOICE DETECTED!",(150,300),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        logger.log_event("SPOOF ATTEMPT ","Speech detected with cloed mouth")
        am.alert_count+=2
    elif is_speech_active and is_mouth_open:
        cv2.putText(frame,"User Speaking..",(20,300),
        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    if am.alert_count> last_alert_count and not is_recording:
        timestamp=time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"evidence/violation_{timestamp}.jpg",frame)
        fourcc=cv2.VideoWriter_fourcc(*'MJPG')
        frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_out=cv2.VideoWriter(f"evidence/clip_{timestamp}.avi",fourcc,20.0,(frame_width,frame_height))
        is_recording=True
        record_timer=60
        print("Evidence capture started:{timestamp}")
    if is_recording:
        video_out.write(frame)
        record_timer-=1
        #Visual indicator that recording is happening
        cv2.circle(frame,(600,30),10,(0,0,255),-1)
        cv2.putText(frame,"REC",(560,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        if record_timer <=0:
            is_recording=False
            video_out.release()
            print("Evidence Capture Saved")
    last_alert_count=am.alert_count
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
    bar_width = int(min(am.current_volume * 200,200))
    cv2.rectangle(frame, (50, 400), (50 + bar_width, 420), (0, 255, 255), -1)
    cv2.putText(frame, "Mic Level", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Audio Warnings: {am.alert_count}", (400, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    if am.alert_count > 10:
        cv2.putText(frame, "CRITICAL: PERSISTENT NOISE", (150, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Master Proctor Dashboard", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()