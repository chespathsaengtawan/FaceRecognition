import face_recognition
import cv2
import warnings
import setuptools
import os
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import io
from PIL import Image

# To avoid pkg_resources deprecation noise, pin setuptools<81 in your requirements (outside this script).
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

# Load employee data from CSV
csv_path = "data/emp.csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Ensure the file exists in data folder.")

# Read employee data
try:
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ['NAME', 'IMAGE_BINARY']):
        raise ValueError("CSV must contain NAME and IMAGE_BINARY columns")
except Exception as e:
    raise RuntimeError(f"Failed to read {csv_path}: {str(e)}")

# Initialize face encodings and names
person_face_encodings = []
person_face_names = []

# Process each employee
for _, row in df.iterrows():
    try:
        # Decode base64 to image
        image_data = base64.b64decode(row['IMAGE_BINARY'])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL image to OpenCV format
        database_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if database_image is None:
            print(f"Warning: Could not decode image for {row['NAME']}")
            continue
            
        # Convert to RGB (face_recognition expects RGB)
        database_image = cv2.cvtColor(database_image, cv2.COLOR_BGR2RGB)
        
        # Ensure array is contiguous
        database_image = np.ascontiguousarray(database_image)
        
        # Get face encodings
        encodings = face_recognition.face_encodings(database_image)
        if len(encodings) > 0:
            person_face_encodings.append(encodings[0])
            person_face_names.append((row['EN'], row['NAME']))
        else:
            print(f"Warning: No face found in image for {row['NAME']}")
            
    except Exception as e:
        print(f"Error processing {row['NAME']}: {str(e)}")
if not person_face_encodings:
    raise RuntimeError("No valid faces found in employee database")

# Video capture
videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    raise RuntimeError("Unable to open webcam (cv2.VideoCapture returned False).")

# Get screen resolution
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

data_locations = []
data_encodings = []
data_names = []
frameProcess = True

while True:
    ret, frame = videoCapture.read()
    if not ret or frame is None:
        break

    # Get screen dimensions
    screen_height, screen_width = frame.shape[:2]
    
    # Resize frame to fit screen while maintaining aspect ratio
    resizing = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    display_frame = cv2.resize(frame, (screen_width, screen_height))

    # convert BGR->RGB and make contiguous uint8 array
    rgb_resizing = resizing[:, :, ::-1]
    rgb_resizing = np.ascontiguousarray(rgb_resizing, dtype=np.uint8)

    if frameProcess:
        data_locations = face_recognition.face_locations(rgb_resizing)
        data_encodings = face_recognition.face_encodings(rgb_resizing, known_face_locations=data_locations)
        data_names = []
        for dc in data_encodings:
            matches = face_recognition.compare_faces(person_face_encodings, dc)
            name_tuple = ("UNKNOWN", "UNKNOWN")
            if True in matches:
                first_match_index = matches.index(True)
                name_tuple = person_face_names[first_match_index]
            data_names.append(name_tuple)

    frameProcess = not frameProcess

    # Create a bottom bar for names
    bottom_bar_height = 60
    bottom_bar = np.zeros((bottom_bar_height, screen_width, 3), dtype=np.uint8)
    
    # Calculate spacing for multiple names
    name_spacing = 20
    current_x = 10    

    # Scale up the coordinates for full screen display
    scale_factor_x = screen_width / (resizing.shape[1] * 4)
    scale_factor_y = screen_height / (resizing.shape[0] * 4)
    
    for (top, right, bottom, left), name_tuple in zip(data_locations, data_names):
        # Scale coordinates for full screen
        top = int(top * 4 * scale_factor_y)
        right = int(right * 4 * scale_factor_x)
        bottom = int(bottom * 4 * scale_factor_y)
        left = int(left * 4 * scale_factor_x)
        
        # Draw face rectangle (green)
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Combine EN and NAME for display
        display_text = f"{name_tuple[0]} : {name_tuple[1]}"
        
        # Calculate text size for dynamic background width
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        
        # Add padding to text width
        padding_x = 20
        bg_width = text_width + (padding_x * 2)
        
        # Draw name in bottom bar (purple background)
        if current_x + bg_width <= screen_width:  # Check if there's space
            # Draw purple background sized to text
            cv2.rectangle(bottom_bar, 
                         (current_x, 0), 
                         (current_x + bg_width, bottom_bar_height),
                         (128, 0, 128),  # Purple
                         cv2.FILLED)
            
            # Add name text (white)
            text_x = current_x + padding_x  # Center text in background
            cv2.putText(bottom_bar, 
                       display_text,
                       (text_x, bottom_bar_height - 20),
                       font,
                       font_scale,
                       (255, 255, 255),  # White
                       thickness)
            
            current_x += bg_width + 10  # Add small gap between names

    # Combine frame with bottom bar
    final_frame = np.vstack([display_frame, bottom_bar])
    
    # Show the combined frame
    cv2.imshow('Video', final_frame)
    
    # Press 'x' to exit, 'f' to toggle fullscreen
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        break
    elif key == ord('f'):
        # Toggle fullscreen
        current_property = cv2.getWindowProperty('Video', cv2.WND_PROP_FULLSCREEN)
        new_property = cv2.WINDOW_NORMAL if current_property == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, new_property)

videoCapture.release()
cv2.destroyAllWindows()