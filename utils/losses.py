import torch
import torch.cuda
import torch.nn.functional as F

def __reduce_loss(loss, mask):
    if mask is None:
        return torch.mean(loss)
    mask_max = (
        mask.view(mask.size(0), -1)
            .amax(1)
            .view(-1, *(1,) * (mask.dim() - 1))
    )
    loss = torch.sum(loss * mask) / torch.sum(mask_max + 1e-8)
    return loss

def weighted_msle_loss(inputs, targets, mask=None):
    loss = (torch.log1p(inputs) - torch.log1p(targets)) ** 2
    return __reduce_loss(loss, mask)

def weighted_mse_loss(inputs, targets, mask=None):
    loss = (inputs - targets) ** 2
    return __reduce_loss(loss, mask)

def weighted_l1_loss(inputs, targets, mask=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    return __reduce_loss(loss, mask)

def weighted_focal_mse_loss(inputs, targets, mask=None, 
                            activate='sigmoid', beta=0.2, gamma=1):
    # Base MSE component
    loss = (inputs - targets) ** 2

    # Apply an activation-based focal-like transformation
    loss = (
        (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma
        if activate == 'tanh'
        else (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) * gamma
    )

    return __reduce_loss(loss, mask)
