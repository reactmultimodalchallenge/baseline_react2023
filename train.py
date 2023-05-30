import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
from dataset import ReactionDataset
from model import TransformerVAE
from utils import AverageMeter
from render import Render
from model.losses import VAELoss
from dataset import get_dataloader

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="./data", type=str, help="dataset path")
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--optimizer-eps', default=1e-8, type=float)
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-seq-len', default=751, type=int, help="length of clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim,', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")

    args = parser.parse_args()
    return args


# Train
def train(model, train_loader, optimizer, criterion):
    losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()

    model.train()
    for batch_idx, (speaker_video_clip, speaker_audio_clip, _, _, _, listener_emotion, listener_3dmm, _) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            speaker_video_clip, speaker_audio_clip, listener_emotion, listener_3dmm = \
                speaker_video_clip[:,:750].cuda(), speaker_audio_clip[:,:750].cuda(), listener_emotion[:,:750].cuda(), listener_3dmm[:,:750].cuda()

        optimizer.zero_grad()
        listener_3dmm_out, listener_emotion_out, distribution = model(speaker_video_clip, speaker_audio_clip)
        loss, rec_loss, kld_loss = criterion(listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out,
                                             distribution)

        losses.update(loss.data.item(), speaker_video_clip.size(0))
        rec_losses.update(rec_loss.data.item(), speaker_video_clip.size(0))
        kld_losses.update(kld_loss.data.item(), speaker_video_clip.size(0))

        loss.backward()
        optimizer.step()
    return losses.avg, rec_losses.avg, kld_losses.avg




# Train
def val(args, model, val_loader, criterion, render, epoch):
    losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()
    model.eval()
    for batch_idx, (speaker_video_clip, speaker_audio_clip, _, _, _, listener_emotion, listener_3dmm, listener_references) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            speaker_video_clip, speaker_audio_clip, listener_emotion, listener_3dmm, listener_references = \
                speaker_video_clip[:,:750].cuda(), speaker_audio_clip[:,:750].cuda(), listener_emotion[:,:750].cuda(), listener_3dmm[:,:750].cuda(), listener_references[:,:750].cuda()

        with torch.no_grad():
            listener_3dmm_out, listener_emotion_out, distribution = model(speaker_video_clip, speaker_audio_clip)

            loss, rec_loss, kld_loss = criterion(listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out, distribution)

            losses.update(loss.data.item(), speaker_video_clip.size(0))
            rec_losses.update(rec_loss.data.item(), speaker_video_clip.size(0))
            kld_losses.update(kld_loss.data.item(), speaker_video_clip.size(0))

            val_path = os.path.join(args.outdir, 'results_videos', 'val')
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            B = speaker_video_clip.shape[0]
            if (batch_idx % 50) == 0:
                for bs in range(B):
                    render.rendering(val_path, "e{}_b{}_ind{}".format(str(epoch + 1), str(batch_idx + 1), str(bs + 1)),
                            listener_3dmm_out[bs], speaker_video_clip[bs], listener_references[bs])


    return losses.avg, rec_losses.avg, kld_losses.avg


def main(args):
    start_epoch = 0
    lowest_val_loss = 10000
    train_loader = get_dataloader(args, "train", load_ref=False, load_video_l=False)
    val_loader = get_dataloader(args, "val", load_video_l=False)
    model = TransformerVAE(img_size = args.img_size, audio_dim = args.audio_dim,  output_3dmm_dim = args._3dmm_dim, output_emotion_dim = args.emotion_dim, feature_dim = args.feature_dim, seq_len = args.seq_len, online = args.online, window_size = args.window_size, device = args.device)
    criterion = VAELoss(args.kl_p)

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.resume != '':
        checkpoint_path = args.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        render = Render('cuda')
    else:
        render = Render()

    for epoch in range(start_epoch, args.epochs):
        train_loss, rec_loss, kld_loss = train(model, train_loader, optimizer, criterion)
        print("Epoch:  {}   train_loss: {:.5f}   train_rec_loss: {:.5f}  train_kld_loss: {:.5f} ".format(epoch+1, train_loss, rec_loss, kld_loss))
        if epoch % 10 == 0:
            val_loss, rec_loss, kld_loss = val(args, model, val_loader, criterion, render, epoch)
            print("Epoch:  {}   val_loss: {:.5f}   val_rec_loss: {:.5f}  val_kld_loss: {:.5f} ".format(epoch+1, val_loss, rec_loss, kld_loss))
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if not os.path.exists(args.outdir):
                    os.makedirs(args.outdir)
                torch.save(checkpoint, os.path.join(args.outdir, 'best_checkpoint.pth'))

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        torch.save(checkpoint, os.path.join(args.outdir, 'cur_checkpoint.pth'))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)

