import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict

st.title("YOLO Object Tracking & Counting")

# Initialize model
model = YOLO("YOLO11l.pt")

# Complete COCO class list (80 classes)
class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Session state to persist counts
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = defaultdict(int)
if 'crossed_ids' not in st.session_state:
    st.session_state.crossed_ids = set()

line_y_red = 430

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
video_path = uploaded_file if uploaded_file else '4.mp4'

if video_path:
    if isinstance(video_path, str):
        # Check if file exists
        import os
        if not os.path.exists(video_path):
            st.error(f"Video file not found at: {video_path}")
            st.stop()
    
    video_placeholder = st.empty()
    counts_placeholder = st.empty()
    
    cap = cv2.VideoCapture(video_path if isinstance(video_path, str) else video_path.name)
    
    if not cap.isOpened():
        st.error("Could not open video file. Please check the format and path.")
        st.stop()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, persist=True, classes=[1,2,3,5,6,])  # Filtering specific classes
        
        if results and results[0].boxes is not None and results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [0]*len(boxes)
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidence = results[0].boxes.conf.cpu()

            cv2.line(frame, (690, line_y_red), (1130, line_y_red), (0, 0, 255), 3)

            for box, tid, class_idx, conf in zip(boxes, track_ids, class_indices, confidence):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_list[class_idx] if class_idx < len(class_list) else str(class_idx)

                    cv2.putText(frame, f"ID: {tid} {class_name}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), radius=4, color=(0, 0, 255), thickness=-1)

                    if center_y > line_y_red and tid not in st.session_state.crossed_ids:
                        st.session_state.crossed_ids.add(tid)
                        st.session_state.class_counts[class_name] += 1
                except Exception as e:
                    st.warning(f"Error processing detection: {e}")
                    continue

        # Display the frame
        video_placeholder.image(frame, channels="BGR")
        
        # Display counts
        counts_text = " | ".join([f"{k}: {v}" for k, v in st.session_state.class_counts.items()])
        counts_placeholder.markdown(f"**Counts:** {counts_text}")
        
    cap.release()
