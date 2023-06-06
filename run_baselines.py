import numpy as np
import torch
import argparse
from tqdm import tqdm
from dataset import get_dataloader
from metric import *
import multiprocessing as mp

baselines = ["GT",
             "Random", "Mime",
             "MeanSeq", "MeanFr", ]
training_mean = None
trainin_mean_single = None

def parse_args():
    parser = argparse.ArgumentParser(description='Running baselines')
    # Param
    parser.add_argument('--dataset-path', default="/home/luocheng/Datasets/S-L", type=str, help="dataset path")
    parser.add_argument('--split', default="val", type=str, help="split of dataset", choices=["val", "test"])
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=751, type=int, help="len of video clip")
    args = parser.parse_args()
    return args

def get_baseline(cfg, baseline, num_pred=10, speaker_emotion=None, listener_emotion=None):
    batch_size = speaker_emotion.shape[0]
    if baseline == "MeanSeq" or baseline == "MeanFr":
        # This baseline predicts the sequence/frame mean of the training data for each emotion dimension.
        global training_mean, training_mean_single
        if training_mean is None or training_mean_single is None:
            train_loader = get_dataloader(cfg, "train", load_emotion_s=True, load_emotion_l=True)
            train_loader._split = "val" # to avoid data augmentation
            all_tr_emotion_list = []
            for batch_idx, (_, _, speaker_emotion, _, _, _, listener_emotion, _, _) in enumerate(tqdm(train_loader)):
                all_tr_emotion_list.append(speaker_emotion.cpu())
                all_tr_emotion_list.append(listener_emotion.cpu())
            
            all_tr_emotion = torch.cat(all_tr_emotion_list, dim = 0)
            # average over all training data
            all_tr_emotion = all_tr_emotion.mean(dim=0)
            single_tr_emotion = all_tr_emotion.mean(dim=0)
            # repeat to match the number of predictions
            training_mean = all_tr_emotion[None, None, ...].repeat(batch_size, num_pred, 1, 1)
            training_mean_single = single_tr_emotion[None, None, ...].repeat(batch_size, num_pred, training_mean.shape[2], 1)

        return training_mean[:batch_size] if baseline == "MeanSeq" else training_mean_single[:batch_size]
    elif baseline == "Random":
        # predict listener emotion as random values between 0 and 1
        return torch.rand(batch_size, num_pred, *speaker_emotion.shape[1:])
    elif baseline == "Mime":
        # predict listener emotion as speaker emotion (mime)
        return speaker_emotion[:, None, ...].repeat(1, num_pred, 1, 1)
    elif baseline == "GT":
        # predict listener emotion as ground truth
        return listener_emotion[:, None, ...].repeat(1, num_pred, 1, 1)
    else:
        raise NotImplementedError("Baseline {} not implemented".format(baseline))


# Train
def val(cfg):
    assert cfg.split in ["val", "test"], "split must be in [val, test]"
    dataloader = get_dataloader(cfg, cfg.split, load_emotion_s=True, load_emotion_l=True, load_audio=False, load_video_s=False, load_video_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False)

    for i, baseline in enumerate(baselines):
        listener_emotion_pred_list = []
        listener_emotion_gt_list = []
        speaker_emotion_list = []
        for batch_idx, (_, _, speaker_emotion, _, _, _, listener_emotion, _, _) in enumerate(tqdm(dataloader)):

            prediction = get_baseline(cfg, baseline, num_pred=10, speaker_emotion=speaker_emotion, listener_emotion=listener_emotion)

            listener_emotion_pred_list.append(prediction)
            listener_emotion_gt_list.append(listener_emotion)
            speaker_emotion_list.append(speaker_emotion)

        all_pred_listener_emotion = torch.cat(listener_emotion_pred_list, dim = 0)
        all_speaker_emotion = torch.cat(speaker_emotion_list, dim = 0)
        all_listener_gt_emotion = torch.cat(listener_emotion_gt_list, dim = 0)

        
        assert all_speaker_emotion.shape[0] == all_pred_listener_emotion.shape[0], "Number of predictions and number of speaker emotions must match ({} vs. {})".format(all_pred_listener_emotion.shape[0], all_speaker_emotion.shape[0])
        #print("-----------------Evaluating Metric-----------------")
        
        p=64
        np.seterr(divide='ignore', invalid='ignore')
        # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
        FRC = compute_FRC_mp(cfg, all_pred_listener_emotion, all_listener_gt_emotion, p=p, val_test=cfg.split)

        # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
        FRD = compute_FRD_mp(cfg, all_pred_listener_emotion, all_listener_gt_emotion, p=p, val_test=cfg.split)

        FRDvs = compute_FRDvs(all_pred_listener_emotion)
        FRVar  = compute_FRVar(all_pred_listener_emotion)
        smse  = compute_s_mse(all_pred_listener_emotion)
        TLCC = compute_TLCC_mp(all_pred_listener_emotion, all_speaker_emotion, p=p)

        # print all results in one line
        print("[{}/{}] Baseline: {}, FRC: {:.5f} | FRD: {:.5f} | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | TLCC: {:.5f}".format(i+1,
                                                                                              len(baselines),
                                                                                              baseline,
                                                                                              FRC,
                                                                                              FRD,
                                                                                              smse,
                                                                                              FRVar,
                                                                                              FRDvs,
                                                                                              TLCC))

        print("Latex-friendly --> B\\_{} & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format(baseline, FRC, FRD, smse, FRVar, FRDvs, TLCC))



def main():
    args = parse_args()
    val(args)


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    main()

