# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS

@MODELS.register_module()
class SKDLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(SKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances."""
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def covariance(self, x, y):
        """Compute the covariance between two sets of features."""
        x_mean = torch.mean(x, dim=0, keepdim=True)
        y_mean = torch.mean(y, dim=0, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        cov = torch.mean(torch.matmul(x_centered.unsqueeze(2), y_centered.unsqueeze(1)), dim=0)
        return cov

    def covariance_loss(self, cov_S, cov_T):
        """Compute the covariance loss."""
        loss = F.mse_loss(cov_S, cov_T)
        return loss

    def forward(self, preds_S: Union[torch.Tensor, Tuple], preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation."""
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            if self.resize_stu:
                pred_S = F.interpolate(pred_S, (pred_T.size(2), pred_T.size(3)), mode='bilinear', align_corners=False)
            assert pred_S.shape == pred_T.shape

            # Apply normalization
            norm_S = self.norm(pred_S)
            norm_T = self.norm(pred_T)

            # Flatten features for covariance calculation
            flat_S = norm_S.flatten(start_dim=2)
            flat_T = norm_T.flatten(start_dim=2)

            # Compute covariance
            cov_S = self.covariance(flat_S, flat_S)
            cov_T = self.covariance(flat_T, flat_T)

            # Compute covariance loss
            loss += self.covariance_loss(cov_S, cov_T)

        return loss * self.loss_weight * 100
