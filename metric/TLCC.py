import torch
import numpy as np
import multiprocessing as mp

def crosscorr(datax, datay, lag=0, dim=25):
    pcc_list = []
    for i in range(dim):
        cn_1, cn_2 = shift(datax[:, i], datay[:, i], lag)
        pcc_i = np.corrcoef(cn_1, cn_2)[0, 1]
        # pcc_i = torch.corrcoef(torch.stack([cn_1, cn_2], dim=0).float())[0, 1]
        pcc_list.append(pcc_i.item())
    return torch.mean(torch.Tensor(pcc_list))


def calculate_tlcc(pred, sp, seconds=2, fps=25):
    rs = [crosscorr(pred, sp, lag, sp.shape[-1]) for lag in range(-int(seconds * fps - 1), int(seconds * fps))]
    peak = max(rs)
    center = rs[len(rs) // 2]
    offset = len(rs) // 2 - torch.argmax(torch.Tensor(rs))
    return peak, center, offset

def compute_TLCC(pred, speaker):
    # pred: N 10 750 25
    # speaker: N 750 25
    offset_list = []
    for k in range(speaker.shape[0]):
        pred_item = pred[k]
        sp_item = speaker[k]
        for i in range(pred_item.shape[0]):
            peak, center, offset = calculate_tlcc(pred_item[i].float().numpy(), sp_item.float().numpy())
            offset_list.append(torch.abs(offset).item())
    return torch.mean(torch.Tensor(offset_list)).item()


def _func(pred_item, sp_item):
    for i in range(pred_item.shape[0]):
        peak, center, offset = calculate_tlcc(pred_item[i], sp_item)
        return torch.abs(offset).item()

def compute_TLCC_mp(pred, speaker, p=8):
    # pred: N 10 750 25
    # speaker: N 750 25
    offset_list = []
    # process each speaker in parallel
    np.seterr(divide='ignore', invalid='ignore')
    
    with mp.Pool(processes=p) as pool:
        # use map
        offset_list += pool.starmap(_func, zip(pred.float().numpy(), speaker.float().numpy()))
    return torch.mean(torch.Tensor(offset_list)).item()


def SingleTLCC(pred, speaker):
    # pred: 10 750 25
    # speaker: 750 25
    offset_list = []
    for i in range(pred.shape[0]):
        peak, center, offset = calculate_tlcc(pred[i].float(), speaker.float())
        offset_list.append(torch.abs(offset).item())
    return torch.mean(torch.Tensor(offset_list)).item()


def shift(x, y, lag):
    if lag > 0:
        return x[lag:], y[:-lag]
    elif lag < 0:
        return x[:lag], y[-lag:]
    else:
        return x, y
