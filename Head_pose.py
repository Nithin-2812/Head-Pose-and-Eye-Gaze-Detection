from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('F:/D-Other_Projects/Codedisha/best(5).pt')


# Open the video file
cap = cv2.VideoCapture(0)

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


# Define the class names
class_names = ['Eye_close', 'Eye_forward', 'Eye_left', 'Eye_right', 'Head_down', 'Head_forward', 'Head_left', 'Head_right']

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference on the frame
    results = model(frame)

    # results.xyxy is a list of detections
    for det in results[0].boxes:
        bbox = det.xyxy[0].int().tolist()  # Convert bounding box coordinates to int
        confidence = det.conf[0].item()
        class_idx = int(det.cls[0].item())
        class_label = class_names[class_idx]  # Get the class name from the class index

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'Class: {class_label}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    # Display the frame with predictions
    cv2.imshow('Frame with Predictions', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()


# Close all OpenCV windows
cv2.destroyAllWindows()
