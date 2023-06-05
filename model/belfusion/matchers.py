
"""
Code adapted from:
https://github.com/BarqueroGerman/BeLFusion
"""

import torch
import torch.nn as nn
import os
import numpy as np
from model.belfusion.mlp_diffae import MLPSkipNet, Activation
from utils import load_config_from_file
import model as module_arch
from model.belfusion.diffusion import LatentDiffusion
from model.belfusion.resample import UniformSampler

class BaseLatentModel(nn.Module):
    def __init__(self, cfg,
                embedder_emotion_path, emb_size=None,
                emb_preprocessing="none",
                freeze_emotion_encoder=True,
                ):
        super(BaseLatentModel, self).__init__()

        self.diffusion = LatentDiffusion(cfg) # TODO init the diffusion object here
        self.schedule_sampler = UniformSampler(self.diffusion)
        
        self.emb_preprocessing = emb_preprocessing
        assert emb_size is not None, "emb_size must be specified."
        self.emb_size = emb_size

        def_dtype = torch.get_default_dtype()

        # load auxiliary model (emotion embedder)
        self.embedder_emotion_path = embedder_emotion_path
        configpath = os.path.join(os.path.dirname(embedder_emotion_path), "config.yaml")
        assert os.path.exists(embedder_emotion_path) and os.path.exists(configpath), f"Missing checkpoint/config file for auxiliary model: '{embedder_emotion_path}'"
        cfg = load_config_from_file(configpath)
        self.embed_emotion = getattr(module_arch, cfg.arch.type)(cfg.arch.args)
    
        checkpoint = torch.load(embedder_emotion_path, map_location='cpu')
        assert "statistics" in checkpoint or emb_preprocessing.lower() == "none", "Model statistics are not available in its checkpoint. Can't apply embeddings preprocessing."
        self.embed_emotion_stats = checkpoint["statistics"] if "statistics" in checkpoint else None
        state_dict = checkpoint['state_dict']
        self.embed_emotion.load_state_dict(state_dict)

        if freeze_emotion_encoder:
            for para in self.embed_emotion.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype) # config loader changes this

        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        # I can't deep copy this class. I need to do this trick to make the deepcopy of everything
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def encode_emotion(self, seq_em):
        return self.preprocess(self.embed_emotion.encode(seq_em))

    def decode_emotion(self, em_emb):
        return self.embed_emotion.decode(self.undo_preprocess(em_emb))
    
    def decode_3dmm(self, reaction):
        return self.embed_emotion.decode_coeff(reaction)

    def get_emb_size(self):
        return self.emb_size

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")
    
    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        self.embed_emotion = self.embed_emotion.to(device)
        if self.embed_emotion_stats is not None:
            for key in self.embed_emotion_stats:
                self.embed_emotion_stats[key] = self.embed_emotion_stats[key].to(device)
        super().to(device)
        return self
    
    def cuda(self):
        return self.to(torch.device("cuda"))
    
    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class LatentMLPMatcher(BaseLatentModel):

    def __init__(self, cfg):
        super(LatentMLPMatcher, self).__init__(cfg, 
                cfg.emb_emotion_path, 
                emb_size=cfg.emb_length, 
                emb_preprocessing=cfg.emb_preprocessing,
                freeze_emotion_encoder=cfg.freeze_emotion_encoder, )

        assert cfg.emb_length is not None, "Embedding length must be specified."
        self.emb_length = cfg.emb_length # TODO multiply by 2 if using speaker + listener embeddings
        self.window_size = cfg.window_size
        assert self.window_size == self.embed_emotion.window_size, "window_size must be the same as the emotion embedder"
        self.online = cfg.online
        
        self.init_params = {
            "num_channels": self.emb_length,
            "skip_layers": cfg.get("skip_layers", "all"),
            "num_hid_channels": cfg.get("num_hid_channels", 2048),
            "num_layers": cfg.get("num_layers", 20),
            "num_time_emb_channels": cfg.get("num_time_emb_channels", 64),
            "num_cond_emb_channels": self.emb_length, # same as num_channels because same embedder is used for speaker and listener
            "activation": cfg.get("activation", Activation.silu),
            "use_norm": cfg.get("use_norm", True),
            "condition_bias": cfg.get("condition_bias", 1),
            "dropout": cfg.get("dropout", 0),
            "last_act": cfg.get("last_act", Activation.none),
            "num_time_layers": cfg.get("num_time_layers", 2),
            "num_cond_layers": cfg.get("num_emotion_layers", 2),
            "time_last_act": cfg.get("time_last_act", False),
            "cond_last_act": cfg.get("cond_last_act", False),
            "dtype": cfg.get("dtype", "float32")
        }
        
        self.model = MLPSkipNet(**self.init_params)

    def forward_offline(self, speaker_emotion=None, listener_emotion=None, **kwargs):
        is_training = self.model.training

        batch_size = speaker_emotion.shape[0]
        if is_training: # sample random subwindow for each batch element
            # sample random window size for all batch elements
            window_start = torch.randint(0, self.window_size, (1,), device=speaker_emotion.device)
            window_end = window_start + self.window_size
            # target to be predicted (and forward diffused)
            x_start = listener_emotion[:, window_start:window_end]
            t, weights = self.schedule_sampler.sample(batch_size, speaker_emotion.device)

            output = self.diffusion.denoise(self, self.model, x_start, t, model_kwargs={
                "cond": speaker_emotion[:, window_start:window_end] # offline mode
            })
            return output
        else: # iterate over all windows
            seq_len = speaker_emotion.shape[1]
            assert seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
            # from [batch_size, seq_len, emotion_dim] to [batch_size * (seq_len // window_length), window_length, emotion_dim]
            diff_batch = batch_size * (seq_len // self.window_size)
            speaker_emotion = speaker_emotion.reshape(diff_batch, self.window_size, -1)
            listener_emotion = listener_emotion.reshape(diff_batch, self.window_size, -1)
            
            output = [output for output in self.diffusion.ddim_sample_loop_progressive(
                self,
                self.model,
                diff_batch,
                model_kwargs={
                    "cond": speaker_emotion,
                },
                gt=listener_emotion
            )][-1] # get last output

            output = {key: output[key].reshape(batch_size, -1, *output[key].shape[2:]) for key in output}
            return output

    def forward_online(self, speaker_emotion=None, listener_emotion=None, **kwargs):
        is_training = self.model.training
        batch_size = speaker_emotion.shape[0]

        first_window_prediction = torch.zeros_like(listener_emotion[:, :self.window_size]) # naive first window prediction
        if is_training:
            # same as offline, but speaker emotion must be shifted by the window size
            # in order to only use past information
            speaker_emotion_shifted = speaker_emotion[:, :-self.window_size]
            listener_emotion_shifted = listener_emotion[:, self.window_size:]
            # for the same listener window to be predicted, the speaker emotion will correspond to the past
            return self.forward_offline(speaker_emotion_shifted, listener_emotion_shifted, **kwargs)

        else:
            # shift speaker emotion by window size and fill with zeros on the left
            # TODO an alternative strategy might be filling it with the most common speaker emotion
            speaker_emotion_shifted = torch.cat([torch.zeros_like(speaker_emotion[:, :self.window_size]), speaker_emotion[:, :-self.window_size]], dim=1)

            return self.forward_offline(speaker_emotion_shifted, listener_emotion, **kwargs)

    def forward(self, **kwargs):
        if self.online:
            return self.forward_online(**kwargs)
        else:
            return self.forward_offline(**kwargs)





if __name__ == '__main__':
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    emb_emotion_path = "out/AutoencoderRNN_W25/230508_154847_812/checkpoint_99.pth"
    
    #device = torch.device('cuda:0')
    device = torch.device('cpu')
    seq_length = 750
    n_features = 25
    batch_size = 64
    emb_length = 126

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "emb_emotion_path": emb_emotion_path,
        "emb_length": emb_length,
        "num_channels": 64,
        "emb_preprocessing": "normalize",
        "freeze_emotion_encoder": True,
    })

    init_params = {
            "num_channels": cfg.emb_length,
            "skip_layers": cfg.get("skip_layers", "all"),
            "num_hid_channels": cfg.get("num_hid_channels", 2048),
            "num_layers": cfg.get("num_layers", 20),
            "num_time_emb_channels": cfg.get("num_time_emb_channels", 65),
            "num_cond_emb_channels": cfg.emb_length,
            "activation": cfg.get("activation", Activation.silu),
            "use_norm": cfg.get("use_norm", True),
            "condition_bias": cfg.get("condition_bias", 1),
            "dropout": cfg.get("dropout", 0),
            "last_act": cfg.get("last_act", Activation.none),
            "num_time_layers": cfg.get("num_time_layers", 2),
            "num_cond_layers": cfg.get("num_emotion_layers", 2),
            "time_last_act": cfg.get("time_last_act", False),
            "cond_last_act": cfg.get("cond_last_act", False),
            "dtype": cfg.get("dtype", "float32")
        }
    model = MLPSkipNet(**init_params)
    print(model)

    # dumb input
    timesteps = torch.tensor(np.array(np.random.randint(0, 1000, batch_size), dtype=int), device=device)

    emb_listener = torch.zeros((batch_size, emb_length), device=device, dtype=dtype).contiguous()
    emb_speaker = torch.zeros((batch_size, emb_length), device=device, dtype=dtype).contiguous()
    print("emb_speaker", emb_speaker.shape, "emb_listener", emb_listener.shape)

    out = model(emb_listener, timesteps, emb_speaker)
    print("\n> OUTPUT:\noutput", out.shape, "\nmu")#, mu.shape, "\nlogvar", logvar.shape)
