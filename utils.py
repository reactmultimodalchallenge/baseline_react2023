from math import cos, pi
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
import os
import yaml


def load_config(config_path=None):
    cli_conf = OmegaConf.from_cli()
    model_conf = OmegaConf.load(cli_conf.pop('config') if config_path is None else config_path)
    return OmegaConf.merge(model_conf, cli_conf)

def load_config_from_file(path):
    return OmegaConf.load(path)

def store_config(config):
    # store config to directory
    dir = config.trainer.out_dir
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(config), f)


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(0, 2, 3, 1)


def torch_img_to_np2(img):
    img = img.detach().cpu().numpy()
    # img = img * np.array([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
    # img = img + np.array([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    img = img * np.array([0.5, 0.5, 0.5]).reshape(1,-1,1,1)
    img = img + np.array([0.5, 0.5, 0.5]).reshape(1,-1,1,1)
    img = img.transpose(0, 2, 3, 1)
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]

    return img


def _fix_image(image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]
    return image

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


