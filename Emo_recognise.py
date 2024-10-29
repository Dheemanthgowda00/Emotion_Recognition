import cv2
from fer import FER

# Initialize the emotion detector
detector = FER()

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(1)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect emotions in the current frame
    emotion_analysis = detector.detect_emotions(frame)

    # Draw bounding boxes and display detected emotions
    for analysis in emotion_analysis:
        # Get bounding box and emotions
        x, y, w, h = analysis['box']
        emotions = analysis['emotions']

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
