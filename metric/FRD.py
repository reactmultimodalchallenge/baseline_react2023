import numpy as np
import os
from tslearn.metrics import dtw
from functools import partial
import multiprocessing as mp


def _func(k_neighbour_matrix, k_pred, em=None):
    neighbour_index = np.argwhere(k_neighbour_matrix == 1).reshape(-1)
    neighbour_index_len = len(neighbour_index)
    min_dwt_sum = 0
    for i in range(k_pred.shape[0]):
        dwt_list = []
        for n_index in range(neighbour_index_len):
            emotion = em[neighbour_index[n_index]]
            res = 0
            for st, ed, weight in [(0, 15, 1 / 15), (15, 17, 1), (17, 25, 1 / 8)]:
                res += weight * dtw(k_pred[i].astype(np.float32)[:, st: ed], emotion.astype(np.float32)[:, st: ed])
            dwt_list.append(res)
        min_dwt_sum += min(dwt_list)
    return min_dwt_sum


def compute_FRD_mp(args, pred, em, val_test='val', p=4):
    # pred: N 10 750 25
    # speaker: N 750 25

    if val_test == 'val':
        neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    else:
        neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))

    FRD_list = []
    with mp.Pool(processes=p) as pool:
        # use map
        _func_partial = partial(_func, em=em.numpy())
        FRD_list += pool.starmap(_func_partial, zip(neighbour_matrix, pred.numpy()))

    return np.mean(FRD_list)



def compute_FRD(args, pred, listener_em, val_test='val'):
    if val_test == 'val':
        speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    else:
        speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))
    all_FRD_list = []
    for i in range(pred.shape[1]):
        FRD_list = []
        for k in range(pred.shape[0]):
            speaker_neighbour_index = np.argwhere(speaker_neighbour_matrix[k] == 1).reshape(-1)
            speaker_neighbour_index_len = len(speaker_neighbour_index)
            dwt_list = []
            for n_index in range(speaker_neighbour_index_len):
                emotion =  listener_em[speaker_neighbour_index[n_index]]
                res = 0
                for st, ed, weight in [(0, 15, 1 / 15), (15, 17, 1), (17, 25, 1 / 8)]:
                    res += weight * dtw(pred[k, i].numpy().astype(np.float32)[:, st: ed],
                                        emotion.numpy().astype(np.float32)[:, st: ed])
                dwt_list.append(res)
            min_dwt = min(dwt_list)
            FRD_list.append(min_dwt)
        all_FRD_list.append(np.mean(FRD_list))
    return sum(all_FRD_list)
