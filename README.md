#  Multi-Detector Traffic Violation System 

## Intro 

- This project uses **YOLOv11/YOLOv8** combined with **BoT-SORT Tracker** to:  
  - Detect and track multiple vehicles in **real traffic videos**.  
  - Identify **traffic light violations**, **wrong-way driving**, and **restricted zone intrusions**.  
  - Assign **unique IDs** to tracked vehicles and log each violation event.  
  - Save cropped images and structured **JSON logs** for each violation.  
  - Display a **real-time mosaic dashboard** combining multiple camera views.  
  - Run all detectors **simultaneously using threads** for smooth performance.
     
---

##   Project Overview

|  **Component** |  **Description** |  **Output** |
|------------------|--------------------|----------------|
|  **TrafficLightDetector** | Detects vehicles running red or yellow lights | JSON log + vehicle snapshot |
|  **ReverseDetector** | Detects vehicles driving in the wrong direction | JSON log + vehicle snapshot |
|  **ZoneDetector** | Detects vehicles entering restricted or no-entry zones | JSON log |
|  **Thread System** | Multi-threaded system for processing multiple cameras in parallel | Real-time mosaic display |
|  **Dashboard Panel** | Displays live statistics and signal status overlay | Directly on the output video |

## Technologies  

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- [BoT-SORT Tracker](https://github.com/NirAharon/BoT-SORT)  
- [OpenCV](https://opencv.org/)  
- [NumPy](https://numpy.org/)  
- Python 3.9 - 3.11
## User Interaction

|  **Action** |  **Description** |
|----------------|-------------------|
|  **Left-click + drag** | Adjust detection lines or zones directly on the video |
|  **Press `S`** | Save the configured lines/zones to a `.npy` file |
|  **Press `ESC`** | Safely exit the program |

##  Run Code 

1. Pip install - r requirements.txt
2.  Python Main.py

##  Output Data Structure

Each output JSON file is named after the **source video** (e.g., `video_name_violations.json`)  
and stores all detected violations with corresponding cropped vehicle images.

Example JSON content:

```json
{
  "id": 12,
  "label": "car",
  "status": "VIOLATION",
  "time": "2025-10-09 12:34:56",
  "image": "violations/cropped_images/car_ID12_1696846492.jpg"
}


