import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model_path = r'C:\Users\hp\Lane_detection\model.h5'
model = load_model(model_path)

# Input and output video paths
vid_input_path = r'C:\Users\hp\Lane_detection\lanes_clip.mp4'
vid_output_path = r'C:\Users\hp\Lane_detection\lanes_output_clip.mp4'


# Function to detect lanes
def detect_lanes(frame):
    # Preprocess frame (resize, normalize, etc. as per your model requirements)
    # Assuming your model expects input shape (height, width, channels)
    input_frame = preprocess(frame)

    # Use the pre-trained model to predict lane markings
    predicted_lanes = model.predict(np.expand_dims(input_frame, axis=0))

    # Post-process predicted lanes (e.g., thresholding, filtering)
    processed_lanes = postprocess(predicted_lanes)

    # Overlay detected lanes on the original frame
    output_frame = overlay(frame, processed_lanes)

    return output_frame


# Function to preprocess frame
def preprocess(frame):
    # Implement your preprocessing steps here
    # Resize, normalize, etc.
    return frame


# Function to post-process predicted lanes
def postprocess(predicted_lanes):
    # Implement any thresholding, filtering, or other post-processing steps
    return predicted_lanes


# Function to overlay detected lanes on the original frame
def overlay(frame, processed_lanes):
    # Implement overlay logic
    # For example, draw the detected lanes on the frame
    # You can use OpenCV functions like cv2.line() to draw lines on the frame
    overlayed_frame = frame  # Placeholder, replace with actual overlay logic
    return overlayed_frame


# Open input video
video_capture = cv2.VideoCapture(vid_input_path)

# Get video properties
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(vid_output_path, fourcc, fps, (width, height))

# Process video frame by frame
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect lanes in the frame
    output_frame = detect_lanes(frame)

    # Write the processed frame to the output video
    video_writer.write(output_frame)

    # Display the processed frame
    cv2.imshow('Lanes Detection', output_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
video_capture.release()
video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
