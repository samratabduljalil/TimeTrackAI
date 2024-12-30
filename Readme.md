# TimeTrackAI  

TimeTrackAI is a Python-based project designed to calculate the average service time for customers at a store's cash counter. The system utilizes video footage, YOLOv8 object detection, and centroid tracking to identify customers and calculate their service times.  

## Features  
- **Bounding Box Detection**: Identifies the specific counter area in the video. 
- **YOLOv8 Integration**: Detects customers in the defined area with high accuracy.  
- **Centroid Tracking**: Tracks customers as they enter and leave the counter area.  
- **Service Time Calculation**: Computes and averages the time customers spend at the counter.  

## How It Works  
1. **Setup**: A bounding box is defined for the cash counter area using video frames .  (hand annotated bbox from  video frame)
2. **Detection**: The YOLOv8 model identifies customers within the bounding box.  
3. **Tracking**: Tracks customer movements and logs entry and exit times.  
4. **Calculation**: Calculates and averages the time each customer spends at the counter.  

## Requirements  
- Python 3.8+  
- OpenCV  
- NumPy  
- Ultralytics  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/TimeTrackAI.git
   cd TimeTrackAI

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Run the script:
   ```bash
   python average_service_time.py

N.B.: This system does not produce 100% accurate results. Further enhancements to detection and tracking algorithms are needed for optimal performance.