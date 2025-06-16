import numpy as np
import streamlit as st
import tempfile
import cv2
import ast
import os

from nanodet.util import Logger
from utils import *
from sort.sort import Sort
from nanodet_detect import NanodetPredictor
from yolo_detect import YoloDetector

from datetime import datetime

SAVE_LOG_DIR = "logs"
SAVE_IMAGE_LOG = "results_images"
logger = Logger(0, use_tensorboard=False)
os.makedirs(SAVE_LOG_DIR, exist_ok=True)
os.makedirs(SAVE_IMAGE_LOG, exist_ok=True)

COLOR_VEHICLE = (0, 255, 255)
COLOR_CLOTHES = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (255, 0, 255),
    3: (255, 255, 0),
    4: (0, 0, 255)
}

CLASS_CLOTHES_IDS = [0, 1, 2, 3, 4]
CLASSES_VEHICLE = [0]  # Motorbike
CLASSES_CLOTHES = {0: 'shirt', 1: 't-shirt', 2: 'sweater',
                   3: 'coat', 4: 'dress'}
SELECTION_TO_VALUE = {
    "LowestAccuracy, BestSpeed": 0,
    "BetterAccuracy, BetterSpeed": 1,
    "BestAccuracy, LowestSpeed": 2
}
CHOICES = list(SELECTION_TO_VALUE.keys())


def app():
    st.set_page_config(layout="wide")
    st.title("Demo video tracking")

    model_option = st.selectbox("Choose Detection Workflow",
                                options=CHOICES)

    choice = SELECTION_TO_VALUE[model_option]

    roi_input = st.text_area("📍 Input ROI points",
                             value="[[1476, 750], [1466, 1340], [2293, 1340], [2289, 750]]")

    vehicle_conf = st.number_input(
        "Vehicle confidence score (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

    clothes_conf = st.number_input(
        "Clothes confidence score (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    video_name = video_file.name.split(".")[0]
    upload_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(
        f"{SAVE_IMAGE_LOG}/{upload_time} {video_name} model_{choice}", exist_ok=True)

    print(upload_time)

    if choice == 0:
        model_detect_vehicle = NanodetPredictor(cfg_path="config/legacy_v0.x_configs/nanodet-m-416-detect-motorbike.yaml",
                                                model_path="workspace/detect_motorbike/nanodet_m_416_detect_motorbike/model_best/model_best.ckpt",
                                                logger=logger,
                                                object="vehicle")

        model_detect_clothes = NanodetPredictor(cfg_path="config/legacy_v0.x_configs/nanodet-m-416-fashionpedia_preprocess.yaml",
                                                model_path="workspace/detect_clothes/nanodet_m_416_FashionPedia_preprocess/model_best/model_best.ckpt",
                                                logger=logger,
                                                object="clothes")
        print("Nanodet - Nanodet workflow")

    elif choice == 1:
        model_detect_vehicle = NanodetPredictor(cfg_path="config/legacy_v0.x_configs/nanodet-m-416-detect-motorbike.yaml",
                                                model_path="workspace/detect_motorbike/nanodet_m_416_detect_motorbike/model_best/model_best.ckpt",
                                                logger=logger,
                                                object="vehicle")
        model_detect_clothes = YoloDetector(model_path="yolo_checkpoint/detect_clothes/FashionPedia_preprocess_320/best.pt",
                                            device="cpu")
        print("Nanodet - YOLO Workflow")
    else:
        model_detect_vehicle = YoloDetector(model_path="yolo_checkpoint/detect_motorbike/Motorbike detection/best.pt",
                                            device="cpu")
        model_detect_clothes = YoloDetector(model_path="yolo_checkpoint/detect_clothes/FashionPedia_preprocess_320/best.pt",
                                            device="cpu")
        print('YOLO - YOLO workflow')

    try:
        wrapped_input = f"{roi_input}"
        point_list = ast.literal_eval(
            wrapped_input)  # kết quả: list các [x, y]
        polygon_pts = np.array(point_list, dtype=np.int32)
    except Exception as e:
        st.error(f"Error when read points {e}")
        st.stop()

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    col1, col2 = st.columns([4, 1])
    video_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    trackers = {cid: Sort() for cid in CLASS_CLOTHES_IDS}

    cap = cv2.VideoCapture(tfile.name)
    frame_w, frame_h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    counted_ids = {cid: set() for cid in CLASS_CLOTHES_IDS}
    total_counts = {cid: 0 for cid in CLASS_CLOTHES_IDS}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if choice == 0:
            preprocess_img = model_detect_vehicle.preprocess(frame)
            results_vehicle = model_detect_vehicle.inference(
                meta=preprocess_img, score_thresh=vehicle_conf)
        elif choice == 1:
            preprocess_img = model_detect_vehicle.preprocess(frame)
            results_vehicle = model_detect_vehicle.inference(
                meta=preprocess_img, score_thresh=vehicle_conf)
        else:
            preprocess_img = frame
            results_vehicle = model_detect_vehicle.predict(
                img=preprocess_img, conf=vehicle_conf)

        detections_per_class = {cid: [] for cid in CLASS_CLOTHES_IDS}

        for result_vehicle in results_vehicle:
            bbox_vehicle = result_vehicle[1:5]

            if is_bbox_partially_inside_polygon(polygon_pts, bbox_vehicle):
                frame = draw_bbox(frame, bbox_vehicle, COLOR_VEHICLE)

                vehicle_img = frame[int(bbox_vehicle[1]):int(bbox_vehicle[3]),
                                    int(bbox_vehicle[0]):int(bbox_vehicle[2])]
                save_img = vehicle_img.copy()
                if choice == 0:
                    preprocess_img_vehicle = model_detect_clothes.preprocess(
                        vehicle_img)
                    result_clothes = model_detect_clothes.inference(
                        meta=preprocess_img_vehicle, score_thresh=clothes_conf)[0] if len(model_detect_clothes.inference(
                            preprocess_img_vehicle, score_thresh=clothes_conf)) else None
                elif choice == 1:
                    preprocess_img_vehicle = vehicle_img
                    result_clothes = model_detect_clothes.predict(
                        img=preprocess_img_vehicle, conf=clothes_conf)[0] if len(model_detect_clothes.predict(
                            vehicle_img, conf=clothes_conf)) else None
                else:
                    preprocess_img_vehicle = vehicle_img
                    result_clothes = model_detect_clothes.predict(
                        preprocess_img_vehicle, conf=clothes_conf)[0] if len(model_detect_clothes.predict(
                            vehicle_img, conf=clothes_conf)) else None

                if result_clothes is not None:
                    bbox_clothes = result_clothes[1:5]
                    clothes_id = result_clothes[0]

                    raw_bbox_clothes = [bbox_clothes[0] + bbox_vehicle[0], bbox_clothes[1] + bbox_vehicle[1],
                                        bbox_clothes[2] + bbox_vehicle[0], bbox_clothes[3] + bbox_vehicle[1]]

                    clothes_img = draw_bbox(
                        save_img, bbox_clothes, COLOR_CLOTHES[int(clothes_id)])
                    clothes_img = draw_text(
                        save_img, f"Conf-{round(float(result_clothes[5]), 2)}", bbox_clothes)

                    if clothes_id in CLASS_CLOTHES_IDS:
                        x1, y1, x2, y2 = map(int, raw_bbox_clothes)
                        detections_per_class[int(clothes_id)].append(
                            [x1, y1, x2, y2, result_clothes[5]])

                for cid in CLASS_CLOTHES_IDS:
                    dets = np.array(detections_per_class[cid]) if len(
                        detections_per_class[cid]) > 0 else np.empty((0, 5))
                    tracks = trackers[cid].update(dets)

                    for track in tracks:
                        x1t, y1t, x2t, y2t, track_id = map(int, track)
                        box = [x1t, y1t, x2t, y2t]

                        if track_id not in counted_ids[cid]:
                            total_counts[cid] += 1
                            counted_ids[cid].add(track_id)

                            if not os.path.exists(f"{SAVE_LOG_DIR}/{upload_time} {video_name} model_{choice}.txt"):
                                with open(f"{SAVE_LOG_DIR}/{upload_time} {video_name} model_{choice}.txt", "w") as log:
                                    time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                                    log.writelines(
                                        f"{time}_{CLASSES_CLOTHES[cid]}_{track_id}\n")
                            else:
                                with open(f"{SAVE_LOG_DIR}/{upload_time} {video_name} model_{choice}.txt", "a") as log:
                                    time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                                    log.writelines(
                                        f"{time}_{CLASSES_CLOTHES[cid]}_{track_id}\n")
                            cv2.imwrite(
                                f"{SAVE_IMAGE_LOG}/{upload_time} {video_name} model_{choice}/{time}_{CLASSES_CLOTHES[cid]}_{track_id}.jpg", clothes_img)

                        color = COLOR_CLOTHES[cid]
                        cv2.rectangle(
                            frame, (x1t, y1t), (x2t, y2t), color, 2)
                        cv2.putText(frame, f'ID{cid}: {track_id}', (x1t, y1t - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        frame = draw_polygon(frame, polygon_pts)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(
            rgb_frame, channels="RGB", use_container_width=True)

        stats_placeholder.markdown(f"""
            ### 📊 Statistic
            - **Shirt:** {total_counts[0]},
            - **T-shirt:** {total_counts[1]},
            - **Sweater:** {total_counts[2]},
            - **Coat:** {total_counts[3]},
            - **Dress:** {total_counts[4]}
        """)

    cap.release()


if __name__ == "__main__":
    app()
