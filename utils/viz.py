from turtle import ht
import cv2
import numpy as np


def visualize_heatmap(htm):
    htm_c = cv2.applyColorMap(np.uint8(htm*255), cv2.COLORMAP_JET)
    return htm_c


def overlay_results(image:np.ndarray, error_map):
    htm = visualize_heatmap(error_map)[..., ::-1]
    if image.dtype is not np.uint8:
        t_image = np.uint8(image*255).copy()
    else:
        t_image = image.copy()
    rlt = cv2.addWeighted(t_image, 0.7, htm, 0.3, 0)
    return rlt, htm