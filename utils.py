import cv2
import numpy as np

from shapely import Polygon, box


def enhance_image(img):

    # img_resized = cv2.resize(img, (320, 640), interpolation=cv2.INTER_LINEAR)

    blurred = cv2.GaussianBlur(img, (9, 9), sigmaX=10)

    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return sharpened


def is_bbox_partially_inside_polygon(polygon_points, bbox):
    poly = Polygon(polygon_points)
    rect = box(*bbox)

    return rect.intersects(poly)


def draw_polygon(image, points):
    points = points.reshape((-1, 1, 2))  # Định dạng cho cv2.polylines

    # Vẽ polygon
    cv2.polylines(image, [points], isClosed=True,
                  color=(0, 0, 255), thickness=2)
    return image


def draw_bbox(image, bbox, color):
    image = cv2.rectangle(
        image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1, cv2.LINE_AA)
    return image


def draw_text(image, text, bbox):
    image = cv2.putText(image, text, (int(bbox[0]), int(bbox[1] + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return image
