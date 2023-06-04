import os
import torch
from torch.utils import data
from torchvision import transforms
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
import time
import pandas as pd
from PIL import Image
import soundfile as sf
import cv2
from torch.utils.data import DataLoader
from multiprocessing import Pool
import torchaudio
from scipy.io import loadmat
torchaudio.set_audio_backend("sox_io")
from functools import cmp_to_key


class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def extract_video_features(video_path, img_transform):
    video_list = []
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = img_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        video_list.append(frame)
    video_clip = torch.cat(video_list, axis=0)
    return video_clip, fps, n_frames


def extract_audio_features(audio_path, fps, n_frames):
    # video_id = osp.basename(audio_path)[:-4]
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs
        # rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        # zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)
        # curr_feat = np.concatenate((curr_mfccs, rms, zcr))

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats


class ReactionDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, root_path, split, img_size=256, crop_size=224, clip_length=751, fps=25,
                 load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True,
                 repeat_mirrored=False):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """

        self._root_path = root_path
        self._img_loader = pil_loader
        self._clip_length = clip_length
        self._fps = fps
        self._split = split

        self._data_path = os.path.join(self._root_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

        self._audio_path = os.path.join(self._data_path, 'Audio_files')
        self._video_path = os.path.join(self._data_path, 'Video_files')
        self._emotion_path = os.path.join(self._data_path, 'Emotion')
        self._3dmm_path = os.path.join(self._data_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)

        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = list(self._list_path.values[:, 1])
        listener_path = list(self._list_path.values[:, 2])

        if self._split in ["train"] or repeat_mirrored:  # training is always mirrored as data augmentation
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp

        self.data_list = []
        for i, (sp, lp) in enumerate(zip(speaker_path, listener_path)):
            ab_speaker_video_path = os.path.join(self._video_path, sp)
            ab_speaker_audio_path = os.path.join(self._audio_path, sp + '.wav')
            ab_speaker_emotion_path = os.path.join(self._emotion_path, sp + '.csv')
            ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + '.npy')

            ab_listener_video_path = os.path.join(self._video_path, lp)
            ab_listener_audio_path = os.path.join(self._audio_path, lp + '.wav')
            ab_listener_emotion_path = os.path.join(self._emotion_path, lp + '.csv')
            ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + '.npy')

            self.data_list.append(
                {'speaker_video_path': ab_speaker_video_path, 'speaker_audio_path': ab_speaker_audio_path,
                 'speaker_emotion_path': ab_speaker_emotion_path, 'speaker_3dmm_path': ab_speaker_3dmm_path,
                 'listener_video_path': ab_listener_video_path, 'listener_audio_path': ab_listener_audio_path,
                 'listener_emotion_path': ab_listener_emotion_path, 'listener_3dmm_path': ab_listener_3dmm_path})

        self._len = len(self.data_list)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================
        changed_sign = 0
        if self._split == 'train':  # only done at training time
            changed_sign = random.randint(0, 1)

        speaker_prefix = 'speaker' if changed_sign == 0 else 'listener'
        listener_prefix = 'listener' if changed_sign == 0 else 'speaker'

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f'{speaker_prefix}_video_path']
        listener_video_path = data[f'{listener_prefix}_video_path']

        if self.load_video_s or self.load_video_l or self.load_ref:  # otherwise, no need to compute these image paths
            img_paths = os.listdir(speaker_video_path)
            total_length = len(img_paths)
            img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
            cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0
            img_paths = img_paths[cp: cp + self._clip_length]

        speaker_video_clip = 0
        if self.load_video_s:
            clip = []
            for img_path in img_paths:
                img = self._img_loader(os.path.join(speaker_video_path, img_path))
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            speaker_video_clip = torch.cat(clip, dim=0)

        # listener video clip only needed for val/test
        listener_video_clip = 0
        if self.load_video_l:
            clip = []
            for img_path in img_paths:
                img = self._img_loader(os.path.join(listener_video_path, img_path))
                img = self._transform(img)
                clip.append(img.unsqueeze(0))
            listener_video_clip = torch.cat(clip, dim=0)


        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio:
            speaker_audio_path = data[f'{speaker_prefix}_audio_path']
            speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)
            speaker_audio_clip = speaker_audio_clip[cp:cp + self._clip_length]


        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        if self.load_emotion_l:
            listener_emotion_path = data[f'{listener_prefix}_emotion_path']
            listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))[
                               cp: cp + self._clip_length]

        if self.load_emotion_s:
            speaker_emotion_path = data[f'{speaker_prefix}_emotion_path']
            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                              cp: cp + self._clip_length]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f'{listener_prefix}_3dmm_path']
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
            listener_3dmm = self._transform_3dmm(listener_3dmm)[0]

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f'{speaker_prefix}_3dmm_path']
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length]
            speaker_3dmm = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:
            img_paths = os.listdir(listener_video_path)
            img_paths = sorted(img_paths, key=cmp_to_key(lambda a, b: int(a[:-4]) - int(b[:-4])))
            listener_reference = self._img_loader(os.path.join(listener_video_path, img_paths[0]))
            listener_reference = self._transform(listener_reference)

        return speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_audio_clip, listener_emotion, listener_3dmm, listener_reference

    def __len__(self):
        return self._len


def get_dataloader(conf, split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=False,
                   load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=False):
    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    # print('==> Preparing data for {}...'.format(split) + '\n')
    dataset = ReactionDataset(conf.dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
                              clip_length=conf.clip_length,
                              load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
                              load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
                              load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)
    shuffle = True if split == "train" else False
    dataloader = DataLoader(dataset=dataset, batch_size=conf.batch_size, shuffle=shuffle, num_workers=conf.num_workers)
    return dataloader
