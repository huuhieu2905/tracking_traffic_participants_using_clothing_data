# QUICK INSTALLATION IN WINDOWS
1. Download and install Docker: https://www.docker.com/get-started/
2. Clone this repository:
   - `git clone https://github.com/huuhieu2905/tracking_traffic_participants_using_clothing_data`
   - `cd tracking_traffic_participants_using_clothing_data`
3. Download checkpoint from this link:
   - NanoDet checkpoint: https://drive.google.com/drive/folders/1a4vinadjP2XigXXqmFeHkNPd8bLvGRc7?usp=drive_link (Download folder workspace)
   - YOLOv11 checkpoint: https://drive.google.com/drive/folders/1a4vinadjP2XigXXqmFeHkNPd8bLvGRc7?usp=drive_link (Download folder yolo_checkpoint)

   Set up folder like this structure:
```
   |-tracking_traffic_participants_using_clothing_data
   |
   |--tracking_app.py
   |--query_app.py
   |.....
   |--workspace/
   |----detect_clothes/
   |----detect_motorbike/
   |--yolo_checkpoint/
   |----detect_clothes/
   |----detect_motorbike/
```
4. Run docker
   - `docker compose up`
5. When docker build complete, access this link:
   - http://localhost:8501 (Tracking app)
   - http://localhost:8502 (Query app)

# TRACKING APP
1. Video demo

[![Demo]](video_demo/tracking_app_demo.mov)


