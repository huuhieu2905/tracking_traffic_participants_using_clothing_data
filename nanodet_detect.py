from nanodet.util.path import mkdir
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline
from nanodet.data.collate import naive_collate
from nanodet.data.batch_process import stack_batch_img
import torch
import cv2
import argparse
import os
import time
import sys


image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]
CLASS_NAME = {
    0: "car",
    1: "motorbike",
    2: "truck",
    3: "bus",
}
MOTORBIKE_ID = 1


class NanodetPredictor(object):
    def __init__(self, cfg_path, model_path, logger, object, device="cpu"):
        load_config(cfg, cfg_path)
        self.object = object
        self.cfg = cfg
        self.device = device
        model = build_model(self.cfg.model)
        ckpt = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)

        if self.cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = self.cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)

        self.model = model.to(device).eval()
        self.pipeline = Pipeline(self.cfg.data.val.pipeline,
                                 self.cfg.data.val.keep_ratio)

    def preprocess(self, img):
        img_info = {"id": 0}

        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(
            meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        return meta

    def inference(self, meta, score_thresh):

        with torch.no_grad():
            results = self.model.inference(meta)
            # results = self.model.inference_speed(meta)

        results = results[0]
        all_box = []

        # Convert result to bounding box
        for label in results:
            for bbox in results[label]:
                score = bbox[-1]
                if score > score_thresh:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    if self.object == "vehicle":
                        if label == MOTORBIKE_ID:
                            all_box.append([label, x0, y0, x1, y1, score])
                    elif self.object == "clothes":
                        all_box.append([label, x0, y0, x1, y1, score])

        all_box.sort(key=lambda v: v[5])

        return all_box


if __name__ == "__main__":
    logger = Logger(0, use_tensorboard=False)
    predict = NanodetPredictor(cfg_path="config/legacy_v0.x_configs/nanodet-m-416-fashionpedia_preprocess.yaml",
                               model_path="workspace/detect_clothes/nanodet_m_416_FashionPedia_preprocess/model_best/model_best.ckpt",
                               logger=logger,
                               object="vehicle",
                               device="cpu")
    img_vehicle = cv2.imread(
        "d:/Code/DATN/DatasetMotorbike/Vietnamese vehicle/test/images/24_jpg.rf.da187b16fd3b098e0dc95f4659dc16e5.jpg")
    img_clothes = cv2.imread(
        "d:/Code/DATN/DatasetClothes/FashionPedia_preprocess/test/images/c67e2bfadf3a9411e77e6ac1fc32a689.jpg")
    pre_img = predict.preprocess(img_clothes)
    predict.inference(pre_img, 0.5)
