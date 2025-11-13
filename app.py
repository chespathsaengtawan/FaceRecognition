import face_recognition
import cv2
import warnings
import requests
import os
import numpy as np
import pandas as pd
from pathlib import Path
import base64
from io import BytesIO
from PIL import ImageTk, Image
import tkinter as tk

# To avoid pkg_resources deprecation noise, pin setuptools<81 in your requirements (outside this script).
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

# API configuration
API_BASE_URL = 'https://api-face-recognition-chi.vercel.app'

# Load employee data from API
def load_employees():
    try:
        response = requests.get(f'{API_BASE_URL}/employees')
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            raise RuntimeError(f"API request failed with status {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch employee data from API: {str(e)}")

def cv2_gui_available():
    try:
        # some builds expose getBuildInformation; quick string check (non-destructive)
        info = getattr(cv2, "getBuildInformation", lambda: "")()
        if "GUI" in info or "GTK" in info or "Win32" in info or "Cocoa" in info:
            # final test: try to create/destroy a window (catch errors)
            try:
                cv2.namedWindow(".__probe__", cv2.WINDOW_NORMAL)
                cv2.destroyWindow(".__probe__")
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False
    
# Initialize face encodings and names
person_face_encodings = []
person_face_names = []

# Process each employee from API
try:
    employees = load_employees()
    for employee in employees:
        try:
            # Decode base64 to image
            image_binary = base64.b64decode(employee['IMAGE_BINARY'])
            image = Image.open(BytesIO(image_binary))
            
            # Convert PIL image to OpenCV format
            database_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if database_image is None:
                print(f"Warning: Could not decode image for {employee['NAME']}")
                continue
                
            # Convert to RGB (face_recognition expects RGB)
            database_image = cv2.cvtColor(database_image, cv2.COLOR_BGR2RGB)
            database_image = np.ascontiguousarray(database_image)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(database_image)
            if len(encodings) > 0:
                person_face_encodings.append(encodings[0])
                person_face_names.append((employee['EN'], employee['NAME']))
            else:
                print(f"Warning: No face found in image for {employee['NAME']}")
                
        except Exception as e:
            print(f"Error processing {employee.get('NAME', 'Unknown')}: {str(e)}")

except Exception as e:
    raise RuntimeError(f"Failed to process employee data: {str(e)}")

if not person_face_encodings:
    raise RuntimeError("No valid faces found in employee database")

# Video capture
videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    raise RuntimeError("Unable to open webcam (cv2.VideoCapture returned False).")

gui_available = cv2_gui_available()
if gui_available:
    try:
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        gui_available = False

if not gui_available:
    # fallback to Tkinter viewer (already in code) — initialize once
    import tkinter as tk
    from PIL import ImageTk, Image
    root = tk.Tk()
    root.title("Video (Tkinter fallback)")
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}")
    root.configure(background='black')
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, highlightthickness=0)
    canvas.pack()
    tk_image_id = None
    _tk_photo = None

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

    if gui_available:
        # Show the combined frame using OpenCV GUI
        try:
            cv2.imshow('Video', final_frame)
        except Exception as e:
            # if imshow suddenly fails, switch to fallback
            print("cv2.imshow failed, switching to Tkinter fallback:", e)
            gui_available = False

            root = tk.Tk()
            root.title("Video (Tkinter fallback)")
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.geometry(f"{screen_w}x{screen_h}")
            root.configure(background='black')
            canvas = tk.Canvas(root, width=screen_w, height=screen_h, highlightthickness=0)
            canvas.pack()
            tk_image_id = None
            _tk_photo = None

    if not gui_available:
        # Convert BGR (OpenCV) to RGB for PIL
        try:
            img_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        except Exception:
            # if conversion fails, try as-is
            img_rgb = final_frame
        pil_img = Image.fromarray(img_rgb)
        # resize to screen while preserving aspect if necessary
        pil_img = pil_img.resize((screen_w, screen_h), Image.LANCZOS)
        _tk_photo = ImageTk.PhotoImage(image=pil_img)
        if tk_image_id is None:
            tk_image_id = canvas.create_image(0, 0, anchor='nw', image=_tk_photo)
        else:
            canvas.itemconfig(tk_image_id, image=_tk_photo)
        # process tkinter events; if window closed, exit loop
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break

    # Press 'x' to exit, 'f' to toggle fullscreen (OpenCV mode only)
    if gui_available:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            break
        elif key == ord('f'):
            current_property = cv2.getWindowProperty('Video', cv2.WND_PROP_FULLSCREEN)
            new_property = cv2.WINDOW_NORMAL if current_property == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, new_property)
# ...existing code...
videoCapture.release()
if not gui_available:
    try:
        root.destroy()
    except Exception:
        pass
cv2.destroyAllWindows()