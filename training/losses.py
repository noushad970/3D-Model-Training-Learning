import torch.nn.functional as F

def voxel_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)
