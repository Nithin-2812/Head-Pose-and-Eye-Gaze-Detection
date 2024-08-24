import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLO model
model = YOLO('F:/D-Other_Projects/Codedisha/Model_best.pt')

# Define the class names
class_names = ['Eye_close', 'Eye_forward', 'Eye_left', 'Eye_right', 'Head_down', 'Head_forward', 'Head_left', 'Head_right']

# Streamlit app layout
st.title("Head Pose and Eye Gaze Detection")
st.write("Upload an image to detect head poses and eye gazes using YOLOv8.")

# Banner Image

st.image('Banner.png', use_column_width=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and process the image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run inference on the image
    results = model(img_array)

    # Convert results to OpenCV format and draw bounding boxes
    for det in results[0].boxes:
        bbox = det.xyxy[0].int().tolist()  # Convert bounding box coordinates to int
        class_idx = int(det.cls[0].item())
        class_label = class_names[class_idx]  # Get the class name from the class index

        # Draw bounding box and label on the frame
        cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
        cv2.putText(img_array, f'{class_label}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (57, 255, 20),2)

    # Convert image to RGB format for Streamlit
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Processed Image', use_column_width=True)