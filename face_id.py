import cv2
from deepface import DeepFace
import os

# Path to your stored ID photo
ID_PHOTO = "id_photo.jpg"

# Check if ID photo exists
if not os.path.exists(ID_PHOTO):
    print(f"Error: {ID_PHOTO} not found! Please add your photo to the folder.")
    exit()

cap = cv2.VideoCapture(0)

print("Press 'v' to verify your face, or 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    display_frame = frame.copy()
    
    # Display Instructions
    cv2.putText(display_frame, "Press 'v' to Verify", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("AI Proctor - Face ID", display_frame)

    key = cv2.waitKey(1)
    if key == ord('v'):
        print("Verifying... please stay still.")
        try:
            # DeepFace Verification logic
            # enforce_detection=False prevents crashing if a face isn't perfectly clear
            result = DeepFace.verify(img1_path = frame, 
                                     img2_path = ID_PHOTO, 
                                     enforce_detection=False)
            
            if result['verified']:
                print("✅ IDENTITY VERIFIED: Match found!")
            else:
                print("❌ ALERT: Identity Mismatch!")
                
        except Exception as e:
            print(f"Verification Error: {e}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()