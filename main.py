import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your lane detection model
model = load_model('C:/Users/hp/Lane_detection/model.h5')


def detect_lanes(image):
    # Your lane detection logic here using the loaded model
    # You can replace this placeholder logic with your actual lane detection code

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest (ROI) for lane detection
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width // 2, height // 2)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw lines on original image
    lane_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combine detected lines with original image
    lane_detected = cv2.addWeighted(image, 0.8, lane_image, 1, 1)

    return lane_detected


# Read input video
video_path = 'C:/Users/hp/Lane_detection/lanes_clip.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
output_path = 'C:/Users/hp/Lane_detection/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect lanes in the frame
    detected_frame = detect_lanes(frame)

    # Write the processed frame to output video
    out.write(detected_frame)

    # Display the processed frame
    cv2.imshow('Lane Detection', detected_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
