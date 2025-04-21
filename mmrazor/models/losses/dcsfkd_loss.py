from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Assuming mmrazor.registry and MODELS are part of the framework you're using
from mmrazor.registry import MODELS

global_image_counter = 0

@MODELS.register_module()
class DCSFKDLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(DCSFKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances."""
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def visualize_weight_matrix(self, att_T, title, filename):
        # 确保att_T已经从Tensor转换为numpy数组并移动到CPU上
        att_T_np = att_T.cpu().detach().numpy()
        # 获取第一张图片的权重矩阵并确保它是一维的
        weights = att_T_np[0, :].squeeze()
        # 确保weights是一维数组
        assert weights.ndim == 1, "Weights must be a 1D array"

        # 创建条形图
        plt.figure(figsize=(12, 6))
        # 绘制条形图
        plt.bar(range(weights.size), weights)
        # 设置x轴和y轴的标签
        plt.xlabel('Channel Index (C)')
        plt.ylabel('Value')
        # 设置y轴的范围
        plt.ylim(1, 1.1)
        # 设置图表标题
        plt.title(title)
        # 保存图表
        plt.savefig(f'{filename}.png')
        plt.close()

        # 使用argsort得到weights的从小到大的索引排序
        sorted_indices = np.argsort(weights)
        # 获取最小的两个值的索引
        min_indices = sorted_indices[:2]
        # 获取最大的两个值的索引
        max_indices = sorted_indices[-2:]

        # 在函数的最后返回最小和最大的两个值的索引
        return min_indices, max_indices



    def visualize_and_save(self, tensor, title, filename, lowchannels, highchannels):
        """可视化张量的前五个通道并保存图像"""
        # 创建一个新的图形
        plt.figure(figsize=(30, 20))  # 设置图形的大小
        plt.suptitle(title)  # 设置总标题

        #显示权重最低和最高的通道for i in range(5):
        plt.subplot(1, 4, 1)
        plt.title(f'Channel {lowchannels[0]}')
        # 显示当前通道的图像
        plt.imshow(tensor[0, lowchannels[0]].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()  # 显示颜色条

        plt.subplot(1, 4, 2)
        plt.title(f'Channel {lowchannels[1]}')
        # 显示当前通道的图像
        plt.imshow(tensor[0, lowchannels[1]].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()  # 显示颜色条

        plt.subplot(1, 4, 3)
        plt.title(f'Channel {highchannels[0]}')
        # 显示当前通道的图像
        plt.imshow(tensor[0, highchannels[0]].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()  # 显示颜色条

        plt.subplot(1, 4, 4)
        plt.title(f'Channel {highchannels[1]}')
        # 显示当前通道的图像
        plt.imshow(tensor[0, highchannels[1]].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()  # 显示颜色条

        # 保存整个图形为一个图片文件
        plt.savefig(f'{filename}.png')
        plt.close()  # 关闭图形，释放资源

    def forward(self, preds_S: Union[torch.Tensor, Tuple], preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        global global_image_counter
        """Forward computation."""
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.0
        index = 0

        for pred_S, pred_T in zip(preds_S, preds_T):
            index += 1
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S != size_T:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear', align_corners=False)
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear', align_corners=False)
            assert pred_S.shape == pred_T.shape

            # Apply normalization
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

            # Compute global average pooling to get channel-wise attentions
            gap_T = F.adaptive_avg_pool2d(norm_T, (1, 1)).view(norm_T.size(0), -1)

            # Apply softmax to the teacher attentions and scale to the range 1-10
            softmax_att_T = F.softmax(gap_T, dim=-1) * 9 + 1  # Scale and shift

            filename3 = 'weight_' + str(index) + str(norm_T.shape) + str(global_image_counter)
            lowchannels, highchannels = self.visualize_weight_matrix(softmax_att_T, 'weight matrix', filename3)

            filename1 = 'norm_T' + str(index) + str(norm_T.shape) + str(global_image_counter)
            self.visualize_and_save(norm_T, 'norm_T', filename1, lowchannels, highchannels)

            # Apply teacher attentions to student feature maps
            att_T = softmax_att_T.view(norm_S.size(0), norm_S.size(1), 1, 1).expand_as(norm_S)

            weighted_S = norm_S * att_T

            # Flatten the feature maps for MSE computation
            weighted_S = weighted_S.view(weighted_S.size(0), weighted_S.size(1), -1)
            norm_T = norm_T.view(norm_T.size(0), norm_T.size(1), -1)
            global_image_counter += 1

            # Compute MSE loss with teacher attentions applied to student feature maps
            loss += F.mse_loss(weighted_S, norm_T)

        return loss * self.loss_weight
