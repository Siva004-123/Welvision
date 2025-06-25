
import cv2
import time
from multiprocessing import Process, Array, Value, Lock, Manager,Queue
from ultralytics import YOLO
import snap7
from snap7.util import set_bool, get_bool
import numpy as np
import os
roller_queue = Queue()

# Shared frame buffer dimensions
frame_shape = (960, 1280, 3)
shared_frame = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())
frame_lock = Lock()

# Global debounce interval
DEBOUNCE_INTERVAL = 0.5

last_detection_time=0



def read_proximity_status(plc_client, byte_index, bool_index):
    """Read proximity sensor status with reconnection logic."""
    try:
        if not plc_client.get_connected():
            plc_client.connect("172.17.8.17", 0, 1)  # Reconnect if disconnected

        data = plc_client.read_area(snap7.type.Areas.DB, 86, 0, 2)
        return get_bool(data, byte_index=byte_index, bool_index=bool_index)
    except Exception as e:
        print(f"Error reading proximity status: {e}")
        return False



def trigger_slot_opening(plc_client, defect_detected):
    try:
        data = bytearray(2)
        if defect_detected:
            set_bool(data, byte_index=1, bool_index=3, value=True)
        else:
            set_bool(data, byte_index=1, bool_index=2, value=True)
        plc_client.write_area(snap7.type.Areas.DB, 86, 0, data)

        # Reset signals after a short delay
    #time.sleep(0.1)
        
        data = bytearray(2)
        set_bool(data, byte_index=1, bool_index=2, value=False)
        set_bool(data, byte_index=1, bool_index=3, value=False)
        plc_client.write_area(snap7.type.Areas.DB, 86, 0, data)
    except Exception as e:
        print(f"Error triggering slot opening: {e}")


def capture_frames(shared_frame, frame_lock):
    """Continuously captureframes from the camera."""
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, -1)
            with frame_lock:
                np_frame = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)
                np.copyto(np_frame, frame)
        else:
            print("Failed to capture frame.")
            time.sleep(0.01)


def process_frames(shared_frame, frame_lock, roller_data, proximity_count,roller_queue):

    """Process frames for YOLO inference and update shared roller_data."""
    detected_folder = "detected_frames"
    os.makedirs(detected_folder, exist_ok=True)
    """Process frames for YOLO inference."""
    # Each process should have its own PLC client
    plc = snap7.client.Client()
    try:
        plc.connect("172.17.8.17", 0, 1)
        print("Process Rollers: Connected to PLC.")
    except Exception as e:
        print(f"Process Rollers: PLC connection error: {e}")
        return  # Exit the process if PLC connection fails
    roller_detected = False
    while True:
        if(read_proximity_status(plc,1,4)) and not roller_detected:
            roller_detected = True
            with frame_lock:
                np_frame = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)
                frame = np_frame.copy()
            pc = 0
            
            proximity_count.value += 1
            pc = proximity_count.value
            # Run YOLO inference
            model_path = r"OldModels\ODlatestmodel.pt"
            yolo = YOLO(model_path)
            results = yolo.predict(frame, device=0, conf=0.2,save=True)
            detections = [
                ("roller" if int(box[-1]) == 4 else "defect", int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                for box in results[0].boxes.data
            ] if results and results[0].boxes.data is not None else []

            # Separate rollers and defects
            roller_boxes = []
            defect_boxes = []

            for bbox in detections:
                class_name, x_min, y_min, x_max, y_max = bbox
                if class_name == "roller":
                    roller_boxes.append((x_min, y_min, x_max, y_max))
                else:
                    defect_boxes.append((class_name, x_min, y_min, x_max, y_max))

            # Sort rollers left to right by x_min
            roller_boxes.sort(key=lambda box: box[0])
            if pc <= 3:
                # Update roller defect status in the shared dictionary
                for i, roller in enumerate(roller_boxes, start=1):
                    roller_id = f"roller_{i}"
                    has_defect = False
                    for defect in defect_boxes:
                        _, x_min, y_min, x_max, y_max = defect
                        # Check intersection
                        if not (roller[2] < x_min or x_max < roller[0] or roller[3] < y_min or y_max < roller[1]):
                            has_defect = True
                            break

                    # Use OR operation to retain defect status across frames
                    roller_data[roller_id] = roller_data.get(roller_id, False) or has_defect
            else:
                for i, roller in enumerate(roller_boxes, start=(pc - 2)):
                    roller_id = f"roller_{i}"
                    has_defect = False
                    for defect in defect_boxes:
                        _, x_min, y_min, x_max, y_max = defect
                        # Check intersection
                        if not (roller[2] < x_min or x_max < roller[0] or roller[3] < y_min or y_max < roller[1]):
                            has_defect = True
                            break

                    # Use OR operation to retain defect status across frames
                    roller_data[roller_id] = roller_data.get(roller_id, False) or has_defect
            if(pc>=3):
                roller_id = f"roller_{pc-2}"
                defect1=roller_data.get(roller_id)
                print("has defect:",defect1)
                roller_queue.put(defect1)

            # Display for debugging
            print("Shared Roller Data:", dict(roller_data))
            print("Proximity Count:", proximity_count.value)
        elif not read_proximity_status(plc, byte_index=1, bool_index=4):
            roller_detected = False
    plc.disconnect()

def handle_slot_control(roller_queue,roller_data):
    """Control slot mechanism based on second proximity sensor."""
    time.sleep(0.1)
    plc = snap7.client.Client()
    try:
        plc.connect("172.17.8.17", 0, 1)
        print("Handle Slot Control: Connected to PLC.")
    except Exception as e:
        print(f"Handle Slot Control: PLC connection error: {e}")
        return  # Exit the process if PLC connection fails
    a=False
    while True:
        if read_proximity_status(plc, byte_index=0, bool_index=2) and not a:
            a=True
            if not roller_queue.empty():
                defect_detected = roller_queue.get()
                print("Trigger:",defect_detected)
                trigger_slot_opening(plc, defect_detected)
                print(f"Processed roller: {'Defective' if defect_detected else 'Good'}")

            # Log the queue state
            queue_size = roller_queue.qsize()
            print(f"Queue size: {queue_size}, Contents: {'Empty' if queue_size == 0 else 'Not Empty'}")

        elif not read_proximity_status(plc, byte_index=0, bool_index=2):
            a=False

    plc.disconnect()

def display_frames(shared_frame, frame_lock):
    """Display frames in a CV2 window."""
    print("Starting frame display...")
    while True:
        with frame_lock:
            np_frame = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)
            frame = np_frame.copy()

        cv2.imshow('Real-Time Frame Display', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Display: Exiting...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Shared resources and initialization
    frame_shape = (960, 1280, 3)
    shared_frame = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())
    frame_lock = Lock()
    manager = Manager()
    roller_data = manager.dict()  # Shared dictionary for roller defect status
    proximity_count = Value('i', 0)
    trigger_count = Value('i', 0)  # Shared integer for proximity count

    # Define processes
    processes = [
        Process(target=capture_frames, args=(shared_frame, frame_lock), daemon=True),
        Process(target=process_frames, args=(shared_frame, frame_lock, roller_data, proximity_count,roller_queue), daemon=True),
        Process(target=handle_slot_control,args=(roller_queue,roller_data), daemon=True)
    ]

    # Start processes
    for process in processes:
        process.start()

    try:
        while True:
            time.sleep(1)  # Main process loop
    except KeyboardInterrupt:
        print("Exiting...")

    # Terminate processes
    for process in processes:
        process.terminate()
        process.join()
