import cv2
import numpy as np
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO

def load_yolo():
    # Load YOLOv8 model
    model = YOLO('D:\\Machine Learning\\my_model\\my_model.pt')
    return model

def detect_objects(img, model, conf_threshold=0.3, nms_threshold=0.4):
    # Run YOLOv8 inference with optimized settings
    results = model(img, conf=conf_threshold, iou=nms_threshold, verbose=False)
    
    boxes_list = []
    confidences = []
    class_ids = []
    
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            boxes_list.append([x1, y1, w, h])
            confidences.append(conf)
            class_ids.append(cls)
    
    return boxes_list, confidences, class_ids, np.arange(len(boxes_list))

def main():
    # Load YOLO
    model = load_yolo()
    
    # Initialize DeepSORT with optimized settings
    tracker = DeepSort(
        max_age=5,
        n_init=1,
        nms_max_overlap=0.7,
        max_cosine_distance=0.3,
        nn_budget=None
    )
    
    # Open video
    cap = cv2.VideoCapture('5.mp4')
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define counting line
    line_y = int(frame_height * 0.5)
    
    # Dictionary to store counts for each class
    class_counts = {name: 0 for name in model.names.values()}
    
    # Dictionary to store tracked objects that have crossed the line
    crossed_objects = set()
    
    # Dictionary to store active tracks
    active_tracks = {}
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = time.time()
        
        # Detect objects
        boxes, confidences, class_ids, indexes = detect_objects(frame, model)
        
        # Prepare detections for DeepSORT
        detections = []
        if len(indexes) > 0:
            for i in range(len(indexes)):
                x, y, w, h = boxes[i]
                detections.append(([x, y, w, h], confidences[i], class_ids[i]))
        
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Clear active tracks for this frame
        current_tracks = set()
        
        # Process tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            current_tracks.add(track_id)
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_id = track.det_class
            class_name = model.names[class_id]
            cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            center_y = (y1 + y2) // 2
            if track_id not in crossed_objects:
                if center_y > line_y:
                    class_counts[class_name] += 1
                    crossed_objects.add(track_id)
        
        # Remove tracks that are no longer active
        for track_id in list(active_tracks.keys()):
            if track_id not in current_tracks:
                del active_tracks[track_id]
        active_tracks = {track_id: track for track in tracks if track.is_confirmed()}
        
        # Display counts and FPS
        y_offset = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            
        # Draw counting line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
        
        # Resize frame
        frame = cv2.resize(frame, (800, 800))
        
        # Display frame
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 