# from scipy.spatial.distance import pdist
import numpy as np
import torch
import os


def compute_s_mse(preds):
    # preds: (B, 10, 750, 25)
    dist = 0
    for b in range(preds.shape[0]):
        preds_item = preds[b]
        if preds_item.shape[0] == 1:
            return 0.0
        preds_item_ = preds_item.reshape(preds_item.shape[0], -1)
        dist_ = torch.pow(torch.cdist(preds_item_, preds_item_), 2)
        dist_ = torch.sum(dist_) / (preds_item.shape[0] * (preds_item.shape[0] - 1) * preds_item_.shape[1])
        dist += dist_
    return dist / preds.shape[0]