'''imported from hongxin001/logitnorm_ood repository'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7 #calcola prima la norma euclidea dei logit lungo l'ultima dimensione del tensore x
        logit_norm = torch.div(x, norms) / self.t # logit vengono normalizzati dividendo ciascun logit per la sua norma, e il risultato viene ulteriormente diviso per l'iperparametro t. Questo passaggio è progettato per regolare l'importanza della normalizzazione dei logit rispetto alla perdita complessiva.
        return F.cross_entropy(logit_norm, target)