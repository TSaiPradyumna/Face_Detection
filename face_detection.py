import cv2
import mediapipe as mp
import time
# Load the video file (Make sure the path is correct)
video_path = "Video/vid1.mp4"
cap = cv2.VideoCapture(video_path)
pTime=0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face_detection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

while True:
    success, img = cap.read()
    #convert the BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            print(id, detection)
            print(detection.score)
            print(detection.location_data.relative_bounding_box)



    # If frame is not read successfully, break the loop (End of video)
    if not success:
        print("End of video or cannot read frame")
        break


    # Resize the frame to a smaller size (e.g., width=640, height=360)
    img_resized = cv2.resize(img, (640, 360))

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime 
    cv2.putText(img_resized,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Video", img_resized)

    # Wait 30ms for next frame; press 'q' to quit
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
