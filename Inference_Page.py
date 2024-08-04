import cv2
import streamlit as st
from ultralytics import YOLO
import psutil
import torch
import time
import pandas as pd
import os

# Load the YOLOv8n model (you can replace 'yolov8n.pt' with your model file)
model = YOLO("best.pt")

def main():
    st.title("Real-time Bearing Detection and Tracking")

    # Sidebar Configuration
    st.sidebar.title("Configuration")

    input_source = st.sidebar.radio(
        "Select input source",
        ('Webcam', 'Local video')
    )

    # Define the classes and initialize their confidence thresholds
    classes = ['chatter_on_od', 'damage_on_big_radius', 'damage_on_od', 'damage_on_small_radius', 'dent', 'roller', 'rust_on_od', 'spherical']
    
    with st.sidebar.expander("Confidence Thresholds"):
        class_thresholds = {cls: st.slider(f"{cls} Confidence Threshold", 0.0, 1.0, 0.5) for cls in classes}

    # Input source selection
    if input_source == "Local video":
        video_file = st.sidebar.file_uploader("Select input video", type=["mp4", "avi"])
        if video_file is not None:
            save_path = os.path.join(os.getcwd(), video_file.name)
            video_path=save_path
            with open(save_path, "wb") as f:
                f.write(video_file.getbuffer())
            
        else:
            st.warning("Please upload a video file.")
            return
    elif input_source == "Webcam":
        video_path = 0  # Default webcam
    print(video_path)  
    # Button to start detection
    start_detection = st.sidebar.button("Start Detection")

    if start_detection:
        # Display system stats
        col1, col2, col3 = st.columns(3)
        with col1:
            cpu_text = st.empty()
        with col2:
            ram_text = st.empty()
        with col3:
            gpu_text = st.empty()

        # Video display
        frame_display = st.empty()

        # Inference stats
        st.subheader("Inference Stats")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.write("FPS")
            fps_chart = st.empty()
        with col5:
            st.write("Total Defective Bearings")
            total_defective_bearings_chart = st.empty()
        with col6:
            st.write("Object Counts")
            object_counts_chart = st.empty()

        # System stats
        st.subheader("System Stats")
        col7, col8, col9 = st.columns(3)
        with col7:
            cpu_usage_text = st.empty()
        with col8:
            ram_usage_text = st.empty()
        with col9:
            gpu_usage_text = st.empty()

        # Video capture
        cap = cv2.VideoCapture(video_path)
        total_defective_bearings = 0
        object_counts = {cls: 0 for cls in classes}
        start_time = time.time()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                results = model(frame)

                # Create a copy of the image for displaying the detections
                output_image = results[0].plot()

                # Update total defective bearings detected and object counts
                for result in results:
                    for detection in result.boxes:
                        confidence = detection.conf[0]
                        class_id = int(detection.cls[0])
                        class_name = model.names[class_id]

                        if class_name in class_thresholds and confidence >= class_thresholds[class_name]:
                            object_counts[class_name] += 1
                            if class_name != 'roller' and class_name != 'spherical':  # Assuming 'roller' and 'spherical' are not defective classes
                                total_defective_bearings += 1

                frame_display.image(output_image, channels="BGR")

                # Calculate FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                # Prepare data for bar charts
                fps_df = pd.DataFrame([fps], columns=["FPS"])
                total_defective_bearings_df = pd.DataFrame([total_defective_bearings/fps], columns=["Total Defective Bearings"])
                object_counts_df = pd.DataFrame(list(object_counts.items()), columns=['Class', 'Count']).set_index('Class')

                # Update inference stats
                fps_chart.bar_chart(fps_df)
                total_defective_bearings_chart.bar_chart(total_defective_bearings_df)
                object_counts_chart.bar_chart(object_counts_df)

                # Update system stats
                cpu_usage_text.text(f"CPU Usage: {psutil.cpu_percent()}%")
                ram_usage_text.text(f"RAM Usage: {psutil.virtual_memory().percent}%")
                if torch.cuda.is_available():
                    gpu_usage_text.text(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
                else:
                    gpu_usage_text.text("GPU not available")

            else:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
