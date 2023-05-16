import torch

def compute_FRVar(preds):
    if len(preds.shape) == 3:
        # preds: (10, 750, 25)
        var = torch.var(preds, dim=1)
        return torch.mean(var)
    elif len(preds.shape) == 4:
        # preds: (N, 10, 750, 25)
        var = torch.var(preds, dim=2)
        return torch.mean(var)
