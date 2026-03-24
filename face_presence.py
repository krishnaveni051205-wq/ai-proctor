import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert to grayscale (Haar Cascades work better in gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_count = len(faces)

    # Logic for Alerts
    status_text = "Status: OK"
    color = (0, 255, 0) # Green

    if face_count == 0:
        status_text = "ALERT: No Face Detected!"
        color = (0, 0, 255) # Red
    elif face_count > 1:
        status_text = f"ALERT: {face_count} Faces Detected!"
        color = (0, 0, 255) # Red

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Display status
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('AI Proctor - OpenCV Version', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()