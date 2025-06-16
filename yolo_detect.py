import cv2

from ultralytics import YOLO


class YoloDetector():
    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, img, conf):
        results = self.model(img, conf=conf, device=self.device)[0]

        all_box = []

        boxes = results.boxes.xyxy
        boxes_id = results.boxes.cls
        boxes_conf = results.boxes.conf

        for box, box_id, box_conf in zip(boxes, boxes_id, boxes_conf):
            box = box.detach().cpu().numpy()
            x0, y0, x1, y1 = int(box[0]), int(
                box[1]), int(box[2]), int(box[3])
            id = box_id
            conf = box_conf

            all_box.append([id, x0, y0, x1, y1, conf])

        all_box.sort(key=lambda v: v[5])

        return all_box


if __name__ == "__main__":
    import time
    img_vehicle = cv2.imread(
        "d:/Code/DATN/DatasetMotorbike/Vietnamese vehicle/test/images/24_jpg.rf.da187b16fd3b098e0dc95f4659dc16e5.jpg")
    img_clothes = cv2.imread(
        "d:/Code/DATN/DatasetClothes/FashionPedia_preprocess/test/images/c67e2bfadf3a9411e77e6ac1fc32a689.jpg")
    predict = YoloDetector(
        model_path="yolo_checkpoint/detect_clothes/FashionPedia_preprocess_320/best.pt")
    time1 = time.time()
    for i in range(1000):
        predict.model(img_clothes)
    time2 = time.time()

    print((time2-time1)/1000)
