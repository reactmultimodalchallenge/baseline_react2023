import numpy as np
import os
from tslearn.metrics import dtw



def compute_FRD(args, pred, em, val_test='val'):
    FRD_list = []
    if val_test == 'val':
        neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    else:
        neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))
    for k in range(pred.shape[0]):
        neighbour_index = np.argwhere(neighbour_matrix[k] == 1).reshape(-1)
        neighbour_index_len = len(neighbour_index)
        dwt_list = []
        for n_index in range(neighbour_index_len):
            emotion = em[neighbour_index[n_index]]
            res = 0
            for st, ed, weight in [(0, 15, 1 / 15), (15, 17, 1), (17, 25, 1 / 8)]:
                res += weight * dtw(pred[k].numpy().astype(np.float32)[:, st: ed], emotion.numpy().astype(np.float32)[:, st: ed])
            dwt_list.append(res)
        min_dwt = min(dwt_list)
        FRD_list.append(min_dwt)

    return np.mean(FRD_list)
