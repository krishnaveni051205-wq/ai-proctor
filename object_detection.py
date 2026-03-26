import cv2
from ultralytics import YOLO
model=YOLO('yolov8n.pt')
UNAUTHORISED=['cell phone','book','laptop']
cap=cv2.VideoCapture(0)
while cap.isOpened():
    sucess,frame=cap.read()
    if not sucess:
        break
    results=model(frame,stream=True,verbose=False,conf=0.5)
    alert_triggered=False
    for r in results:
        for box in r.boxes:
            cls_id=int(box.cls[0])
            class_name=model.names[cls_id]
            if class_name in UNAUTHORISED:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,f"Alert :{class_name}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                alert_triggered=True
    status_text="STATUS: UNAUTHORISED OBJECTS!" if alert_triggered else "STATUS:CLEAN"
    color=(0,0,255) if alert_triggered else (0,255,0)
    cv2.putText(frame,status_text,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    cv2.imshow("AI_PROCTOR- Object detection",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()