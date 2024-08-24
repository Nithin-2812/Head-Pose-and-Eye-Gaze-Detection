# Head Pose and Eye Gaze Detection Web App
### Overview
This web application leverages deep learning to accurately detect and classify head poses and eye gazes in real-time. It is built using the YOLOv8 nano model, a state-of-the-art object detection framework. The model was trained on a custom dataset with 1041 training images and 260 validation images, annotated for compatibility with YOLOv8. The application allows users to upload images, process them, and visualize the detected head poses and eye gazes directly in their web browser.
<br/>

## Demo Video

<p align="center">
  <img src="https://github.com/Nithin-2812/Head-Pose-and-Eye-Gaze-Detection/blob/main/head-pose-detect_demo.gif" alt="animated" />
</p>

## Features

1. <b>Real-Time Detection: </b>Detects and classifies various head poses and eye gaze directions in uploaded images.

2. <b>User-Friendly Interface: </b>Simple and intuitive web interface built using Streamlit.

3. <b>High Accuracy: </b>Powered by the YOLOv8 nano model, trained for 150 epochs on a specialized dataset, ensuring reliable predictions.

4. <b>Custom Dataset: </b>Utilizes a dataset of 1041 training images and 260 validation images, meticulously annotated for precise head pose and eye gaze detection.

## Technology Stack

1. <b>YOLOv8 Nano Model: </b>A lightweight yet powerful deep learning model for object detection.
  
2. <b>Python: </b>Core programming language used to build the application.

3. <b>Streamlit: </b>Framework used to create the interactive web interface.

4. <b>OpenCV: </b>Library for image processing and handling webcam input.

5. <b>NumPy: </b>Used for efficient numerical computations.

6. <b>Pillow: </b>Library for image handling and processing.

7. <b>Google </b>Colab: Platform used for training the YOLOv8 nano model on the custom dataset.

8. <b>GitHub: </b>Version control and code hosting platform used to manage and share the project.


## Installation

1. **Clone the Repository:**

    ```bash
    https://github.com/Nithin-2812/Head-Pose-and-Eye-Gaze-Detection.git
    cd Head-Pose-and-Eye-Gaze-Detection.git
    ```

2. **Install Required Packages:**

    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```
