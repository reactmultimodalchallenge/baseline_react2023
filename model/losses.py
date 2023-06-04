import torch
import torch.nn as nn
import torch.nn.functional as F



class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"



class VAELoss(nn.Module):
    def __init__(self, kl_p=0.0002):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, gt_emotion, gt_3dmm, pred_emotion, pred_3dmm, distribution):
        rec_loss = self.mse(pred_emotion, gt_emotion) + self.mse(pred_3dmm[:,:, :52], gt_3dmm[:,:, :52]) + 10*self.mse(pred_3dmm[:,:, 52:], gt_3dmm[:,:, 52:])
        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss


        return loss, rec_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"



def div_loss(Y_1, Y_2):
    loss = 0.0
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss
