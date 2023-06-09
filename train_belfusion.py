import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
import model as module_arch
from metric import *
import model.losses as module_loss
from functools import partial
from utils import load_config, store_config, AverageMeter
from dataset import get_dataloader
import wandb
from datetime import datetime
import random
from omegaconf import OmegaConf


def evaluate(cfg, pred_list_em, speaker_em, listener_em, epoch):
    """
    Must receive
        pred_list_em: [dataset_size, num_preds, seq_length, 25]
        speaker_em: [dataset_size, seq_length, 25]
        listener_em: [dataset_size, seq_length, 25]
    """
    assert listener_em.shape[0] == speaker_em.shape[0], "speaker and listener emotion must have the same shape"
    assert listener_em.shape[0] == pred_list_em.shape[0], "predictions and listener emotion must have the same shape"

    # only the fast diversity metrics ploted often
    metrics = {
        # APPROPRIATENESS METRICS
        #"FRDist": compute_FRD(data_path, pred_list_em[:,0], listener_em), # FRDist (1) --> slow, ~3 mins
        #"FRCorr": compute_FRC(data_path, pred_list_em[:,0], listener_em), # FRCorr (2) --> slow, ~3 mins

        # DIVERSITY METRICS --> all very fast, compatible with validation in training loop
        "FRVar": compute_FRVar(pred_list_em), # FRVar (1) --> intra-variance (among all frames in a prediction),
        "FRDiv": compute_s_mse(pred_list_em), # FRDiv (2) --> inter-variance (among all predictions for the same speaker),
        "FRDvs": compute_FRDvs(pred_list_em), # FRDvs (3) --> diversity among reactions generated from different speaker behaviours
        
        # OTHER METRICS
        # FRRea (realism)
        #"FRSyn": compute_TLCC(pred_list_em, speaker_em), # FRSyn (synchrony) --> EXTREMELY slow, ~1.5h
    }
    return metrics


def update_averagemeter_from_dict(results, meters):
    # if meters is empty, it will be initialized. If not, it will be updated
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            value = value.item()

        if key in meters:
            meters[key].update(value)
        else:
            meters[key] = AverageMeter()
            meters[key].update(value)

# Train
def train(cfg, model, train_loader, optimizer, criterion, device):
    losses_meters = {}
    model.train()
    for batch_idx, (_, _, s_emotion, _, _, _, l_emotion, l_3dmm, l_reference) in enumerate(tqdm(train_loader)):
        s_emotion, l_emotion, l_3dmm = s_emotion.to(device), l_emotion.to(device), l_3dmm.to(device)
        
        optimizer.zero_grad()

        prediction = model(speaker_emotion=s_emotion, listener_emotion=l_emotion, listener_3dmm=l_3dmm) # returns dictionary with "prediction" and "target"
        assert "target" in prediction, "target must be in prediction"

        losses = criterion(**prediction)
        update_averagemeter_from_dict(losses, losses_meters)
        losses["loss"].backward()
        optimizer.step()

    return {key: losses_meters[key].avg for key in losses_meters}


def validate(cfg, model, val_loader, criterion, device, epoch, mode):
    num_preds = cfg.trainer.get("num_preds", 10) # number of predictions to make
    losses_meters = {}
    model.eval()
    #model.train() # set model to train mode to simulate training
    #num_preds = 1
    with torch.no_grad():
        all_predictions, speaker_emotions, listener_emotions = [], [], []
        for batch_idx, (_, _, s_emotion, _, _, _, l_emotion, l_3dmm, l_reference) in enumerate(tqdm(val_loader)):
            # make num_preds predictions by repeating the same input --> same batch to speed up inference
            s_emotion_, l_emotion_, l_3dmm_ = s_emotion.to(device), l_emotion.to(device), l_3dmm.to(device)
            if mode == "predict": # then predict num_preds
                s_emotion_ = s_emotion_.repeat_interleave(num_preds, dim=0) # [num_preds*batch_size, seq_length, 25]
                l_emotion_ = l_emotion_.repeat_interleave(num_preds, dim=0) # [num_preds*batch_size, seq_length, 25]
                l_3dmm_ = l_3dmm_.repeat_interleave(num_preds, dim=0) # [num_preds*batch_size, seq_length, 25]
            
            prediction = model(speaker_emotion=s_emotion_, listener_emotion=l_emotion_, listener_3dmm=l_3dmm_) # [num_preds*batch_size, seq_length, 25]
            assert "target" in prediction, "target must be in prediction"

            if mode == "predict":
                # prediction needs to be transformed from [num_preds*batch_size, seq_length, 25] to [batch_size, num_preds, seq_length, 25]
                prediction = {key: prediction[key].view(-1, num_preds, *prediction[key].shape[1:]) for key in prediction}

            losses = criterion(**prediction)
            update_averagemeter_from_dict(losses, losses_meters)

            # save predictions
            all_predictions.append(prediction["prediction"])
            speaker_emotions.append(s_emotion)
            listener_emotions.append(l_emotion)

        # compute metrics
        if mode == "predict":
            # concatenate all predictions
            prediction = torch.cat(all_predictions, dim=0) # [num_preds*dataset_size, num_preds, seq_length, 25]
            speaker_emotion = torch.cat(speaker_emotions, dim=0) # [num_preds*dataset_size, seq_length, 25]
            listener_emotion = torch.cat(listener_emotions, dim=0) # [num_preds*dataset_size, seq_length, 25]

            metrics_results = evaluate(cfg, prediction.cpu(), speaker_emotion.cpu(), listener_emotion.cpu(), epoch)
        else:
            metrics_results = {} # no metrics computed for autoencode mode

    return {"val_" + key: losses_meters[key].avg for key in losses_meters}, metrics_results

def compute_statistics(config, model, data_loader, device):
    checkpoint_path = config.resume
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reload checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_idx, (_, _, speaker_emotion, _, _, _, listener_emotion, _, _) in enumerate(tqdm(data_loader)):
            speaker_emotion, listener_emotion = speaker_emotion.to(device), listener_emotion.to(device)
            prediction = model.encode_all(speaker_emotion) # [num_windows * batch_size, features_size]
            preds.append(prediction)
            prediction = model.encode_all(listener_emotion) # [num_windows * batch_size, features_size]
            preds.append(prediction)

    preds = torch.cat(preds, axis=0)

    checkpoint["statistics"] = {
        "min": preds.min(axis=0).values,
        "max": preds.max(axis=0).values,
        "mean": preds.mean(axis=0),
        "std": preds.std(axis=0),
        "var": preds.var(axis=0),
    }
    
    torch.save(checkpoint, config.resume)

def main():
    # load yaml config
    cfg = load_config()
    cfg.trainer.out_dir = os.path.join(cfg.trainer.out_dir, cfg["name"])
    os.makedirs(cfg.trainer.out_dir, exist_ok=True)
    store_config(cfg)

    start_epoch = 0
    train_loader = get_dataloader(cfg.dataset, cfg.dataset.split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=True,
                   load_emotion_l=True, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=False)
    valid_loader = get_dataloader(cfg.validation_dataset, cfg.validation_dataset.split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=True,
                   load_emotion_l=True, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(module_arch, cfg.arch.type)(cfg.arch.args)
    model = model.to(device)
    print('Model {} : params: {:4f}M'.format(cfg.arch.type, sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    if cfg.trainer.resume != None:
        checkpoint_path = cfg.trainer.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)

    mode = cfg.trainer.get("mode", "predict")
    assert mode in ["autoencode", "predict"], "mode must be either autoencode or predict"
    print("WARNING: mode is set to {}".format(mode))
    last_epoch_stored = 0
    val_loss = 0
    val_metrics = None
    log_dict = {}
    for epoch in range(start_epoch, cfg.trainer.epochs):
        log_dict["lr"] = optimizer.param_groups[0]['lr']
        log_dict["epoch"] = epoch

        # =================== TRAIN ===================
        train_losses = train(cfg, model, train_loader, optimizer, criterion, device)
        log_dict.update(train_losses)

        # =================== VALIDATION ===================
        if (cfg.trainer.val_period > 0 and (epoch + 1) % cfg.trainer.val_period == 0) or epoch == start_epoch:
            val_losses, val_metrics = validate(cfg, model, valid_loader, criterion, device, epoch, mode)
            log_dict.update(val_losses)
            log_dict.update(val_metrics)

        # =================== log ===================
        log_message = 'epoch: {}'.format(epoch)
        for key, value in log_dict.items():
            log_message += ", {}: {:.3f}".format(key, value)
        print(log_message)

        # =================== save checkpoint ===================
        if (cfg.trainer.save_period > 0 and (epoch + 1) % cfg.trainer.save_period == 0):
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            os.makedirs(cfg.trainer.out_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(cfg.trainer.out_dir, 'checkpoint_{}.pth'.format(epoch)))
            # remove older
            older_path = os.path.join(cfg.trainer.out_dir, 'checkpoint_{}.pth'.format(epoch - cfg.trainer.save_period))
            if os.path.exists(older_path):
                os.remove(older_path)
            last_epoch_stored = epoch

    if mode == "autoencode":
        # =================== Compute STATISTICS ===================
        print(f"Starting stats computation...")
        cfg.resume = os.path.join(cfg.trainer.out_dir, f"checkpoint_{last_epoch_stored}.pth")
        compute_statistics(cfg, model, train_loader, device)
        print("Stats computed!")
        print('=' * 80)


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    # set all seeds to 6
    torch.manual_seed(6)
    torch.cuda.manual_seed(6)
    np.random.seed(6)
    random.seed(6)
    main()

