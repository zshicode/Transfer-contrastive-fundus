import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.pool1 = nn.AvgPool2d(2) # 16x16 -> 8x8
        self.lin1 = nn.Linear(64,d) # 64 -> 384
        self.conv = nn.Sequential(
            self.conv1,self.pool1,nn.Tanh(),
            nn.Flatten(),self.lin1
        )
        self.lin2 = nn.Linear(d,256) # 384 -> 256
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        
    def forward(self,x):
        x = self.conv(x)
        z = x.squeeze()
        x = self.lin2(x)
        x = x.reshape(-1,1,16,16)
        x = self.conv2(x)
        return x,z

class DiceLossMultiClass(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]  class indices
        """
        C = logits.size(1)
        probs = F.softmax(logits, dim=1)   # [B,C,H,W]

        # one-hot
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0,3,1,2).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)
            probs = probs * mask
            targets_onehot = targets_onehot * mask

        # flatten
        probs = probs.contiguous().view(probs.size(0), C, -1)
        targets_onehot = targets_onehot.contiguous().view(targets_onehot.size(0), C, -1)

        intersection = (probs * targets_onehot).sum(dim=-1)
        union = probs.sum(dim=-1) + targets_onehot.sum(dim=-1)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()
        return dice_loss

def get_pred(logits):
    """ logits [B,C,H,W] -> pred [B,H,W] """
    return torch.argmax(logits, dim=1)

def dice_coeff(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index=None):
    """
    Dice coefficient, macro average
    pred, target: [B, H, W]
    return: (per_class_dice, mean_dice)
    """
    pred = pred.long()
    target = target.long()
    smooth = 1e-6
    dice_list = []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        p = (pred == c).float()
        t = (target == c).float()

        inter = (p * t).sum()
        union = p.sum() + t.sum()
        d = (2.0 * inter + smooth) / (union + smooth)
        dice_list.append(d.item())
    return dice_list, np.mean(dice_list) if len(dice_list) > 0 else 0.0

def recall_coeff(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index=None):
    """
    Recall = TP / (TP + FN)
    """
    pred = pred.long()
    target = target.long()
    smooth = 1e-6
    recall_list = []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        p = (pred == c).float()
        t = (target == c).float()

        tp = (p * t).sum()
        fn = ((1 - p) * t).sum()
        r = (tp + smooth) / (tp + fn + smooth)
        recall_list.append(r.item())
    return recall_list, np.mean(recall_list) if len(recall_list) > 0 else 0.0

def hausdorff_95(pred_np: np.ndarray, target_np: np.ndarray):
    """
    single binary image HD95
    pred_np, target_np: (H,W) bool array
    """
    pred_pts = np.argwhere(pred_np)
    tgt_pts = np.argwhere(target_np)
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        return np.nan

    d1 = directed_hausdorff(pred_pts, tgt_pts)[0]
    d2 = directed_hausdorff(tgt_pts, pred_pts)[0]
    all_dist = []
    def dist_set(A, B):
        from scipy.spatial import cKDTree
        tree = cKDTree(B)
        dist, _ = tree.query(A, k=1)
        return dist
    dA = dist_set(pred_pts, tgt_pts)
    dB = dist_set(tgt_pts, pred_pts)
    all_dist = np.concatenate([dA, dB])
    hd95 = np.percentile(all_dist, 95)
    return hd95

def hd95_all_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index=None):
    """
    pred, target: [B, H, W]
    """
    pred = pred.long().cpu().numpy()
    target = target.long().cpu().numpy()
    hd_per_class = [[] for _ in range(num_classes)]

    B = pred.shape[0]
    for b in range(B):
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            p_mask = pred[b] == c
            t_mask = target[b] == c
            hd = hausdorff_95(p_mask, t_mask)
            if not np.isnan(hd):
                hd_per_class[c].append(hd)
    
    class_mean = []
    for c in range(num_classes):
        if len(hd_per_class[c]) > 0:
            class_mean.append(np.mean(hd_per_class[c]))
        else:
            class_mean.append(np.nan)
    valid = [x for x in class_mean if not np.isnan(x)]
    mean_hd95 = np.mean(valid) if valid else np.nan
    return class_mean, mean_hd95

def compute_metrics(logits, target, num_classes, ignore_index=None):
    pred = get_pred(logits)
    _, mdice = dice_coeff(pred, target, num_classes, ignore_index)
    _, mrecall = recall_coeff(pred, target, num_classes, ignore_index)
    _, mhd95 = hd95_all_class(pred, target, num_classes, ignore_index)
    return {
        "dice": mdice,
        "recall": mrecall,
        "hd95": mhd95
    }