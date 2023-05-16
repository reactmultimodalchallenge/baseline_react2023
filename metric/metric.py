import torch
import numpy as np

def s_mse(preds):
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


def FRVar(preds):
    if len(preds.shape) == 3:
        # preds: (10, 750, 25)
        var = torch.var(preds, dim=1)
        return torch.mean(var)
    elif len(preds.shape) == 4:
        # preds: (N, 10, 750, 25)
        var = torch.var(preds, dim=2)
        return torch.mean(var)



def FRDvs(preds):
    # preds: (N, 10, 750, 25)
    preds_ = preds.reshape(preds.shape[0], preds.shape[1], -1)
    preds_ = preds_.transpose(0, 1)
    # preds_: (10, N, 750*25)
    dist = torch.pow(torch.cdist(preds_, preds_), 2)
    # dist: (10, N, N)
    dist = torch.sum(dist) / (preds.shape[0] * (preds.shape[0] - 1) * preds.shape[1])
    return dist / preds_.shape[-1]


from scipy.spatial.distance import pdist
import numpy as np
import torch
import os

def compute_FRVar(pred):
    FRVar_list = []
    for k in range(pred.shape[0]):
        pred_item = pred[k]
        for i in range(0, pred_item.shape[0]):
            var = np.mean(np.var(pred_item[i].numpy().astype(np.float32), axis=0))
            FRVar_list.append(var)
    return np.mean(FRVar_list)



