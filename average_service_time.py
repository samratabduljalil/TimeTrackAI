import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure to download YOLOv8 model weights

# Video input
video_path = "fringestorez.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()
print("Work in Progress ......")
# Bounding box coordinates from the XML annotation
xmin = 867
ymin = 66
xmax = 1250
ymax = 718

# Get the frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define tracking and timing structures
customer_timings = {}
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Parameters for filtering
CONFIDENCE_THRESHOLD = 0.80 # Minimum confidence for detections
MIN_BBOX_AREA = 500  # Minimum bounding box area to consider as a customer

# Centroid tracker
trackers = {}
next_customer_id = 0
MAX_DISAPPEARED_FRAMES = 50

frame_count = 0

def compute_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def match_existing_customers(trackers, centroids, max_distance=50):
    updated_trackers = {}
    unmatched_centroids = set(range(len(centroids)))

    for customer_id, data in trackers.items():
        last_centroid, disappeared_frames = data
        distances = [np.linalg.norm(np.array(last_centroid) - np.array(c)) for c in centroids]
        if distances and min(distances) < max_distance:
            matched_idx = np.argmin(distances)
            updated_trackers[customer_id] = (centroids[matched_idx], 0)
            unmatched_centroids.discard(matched_idx)
        else:
            if disappeared_frames < MAX_DISAPPEARED_FRAMES:
                updated_trackers[customer_id] = (last_centroid, disappeared_frames + 1)

    return updated_trackers, unmatched_centroids

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame using the bounding box
    cropped_frame = frame[ymin:ymax, xmin:xmax]

    # Run YOLO model on the cropped frame
    results = model(cropped_frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    centroids = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection

        # Filter by confidence and 'person' class (class id 0 for YOLO trained on COCO dataset)
        if conf >= CONFIDENCE_THRESHOLD and int(cls) == 0:
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height

            # Filter by bounding box size
            if bbox_area >= MIN_BBOX_AREA:
                centroids.append(compute_centroid((x1, y1, x2, y2)))

    # Update trackers
    trackers, unmatched_centroids = match_existing_customers(trackers, centroids)

    # Assign new IDs to unmatched centroids
    for idx in unmatched_centroids:
        trackers[next_customer_id] = (centroids[idx], 0)
        customer_timings[next_customer_id] = {"entry_frame": frame_count, "exit_frame": None}
        next_customer_id += 1

    # Update exit frames for disappeared customers
    for customer_id, (_, disappeared_frames) in trackers.items():
        if disappeared_frames >= MAX_DISAPPEARED_FRAMES and customer_timings[customer_id]["exit_frame"] is None:
            customer_timings[customer_id]["exit_frame"] = frame_count

    frame_count += 1

cap.release()

# Calculate service times
service_times = []
for customer_id, timing in customer_timings.items():
    entry_time = timing["entry_frame"] / fps
    exit_time = timing["exit_frame"] / fps if timing["exit_frame"] else None

    if exit_time and (exit_time - entry_time) > 2:
        service_times.append(exit_time - entry_time)
      

# Calculate average service time
if service_times:
    average_service_time = sum(service_times) / len(service_times)
    print(f"Average service time: {average_service_time:.2f} seconds")
else:
    print("No valid customer service times detected.")
