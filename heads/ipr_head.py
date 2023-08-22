# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)
from mmpose.core.post_processing import fliplr_regression
from mmpose.models.builder import HEADS, build_loss


@HEADS.register_module()
class IntegralPoseRegressionHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_joints,
                 feat_size,
                 loss_keypoint=None,
                 out_sigma=False,
                 debias=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.out_sigma = out_sigma
        self.debias = debias

        self.conv = build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=in_channels,
                            out_channels=num_joints,
                            kernel_size=1,
                            stride=1,
                            padding=0)

        self.size = feat_size
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1]) / self.size
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size]) / self.size
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

        if out_sigma:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.in_channels, self.num_joints * 2)
        # if debias:
        #     self.softmax_fc = nn.Linear(64, 64)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, ('DeepPoseRegressionHead only supports '
                                 'single-level feature.')
            x = x[0]

        featmap = self.conv(x)
        s = list(featmap.size())
        featmap = featmap.view([s[0], s[1], s[2] * s[3]])
        # if self.debias:
        #     mlp_x_norm = torch.norm(self.softmax_fc.weight, dim=-1)
        #     norm_feat = torch.norm(featmap, dim=-1, keepdim=True)
        #     featmap = self.softmax_fc(featmap)
        #     featmap /= norm_feat
        #     featmap /= mlp_x_norm.reshape(1, 1, -1)
        #     featmap *= self.beta
            
        featmap = F.softmax(featmap, dim=2)
        featmap = featmap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = featmap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2, keepdim=True)
        scoremap_y = featmap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2, keepdim=True)
            
        if self.debias:
            C = featmap.reshape(s[0], s[1], s[2] * s[3]).exp().sum(dim=2).unsqueeze(dim=2)
            soft_argmax_x = C / (C - 1) * (soft_argmax_x - 1 / (2 * C))
            soft_argmax_y = C / (C - 1) * (soft_argmax_y - 1 / (2 * C))
            
        output = torch.cat([soft_argmax_x, soft_argmax_y], dim=-1)
        if self.out_sigma:
            x = self.gap(x).reshape(x.size(0), -1)
            pred_sigma = self.fc(x)
            pred_sigma = pred_sigma.reshape(pred_sigma.size(0), self.num_joints, 2)
            output = torch.cat([output, pred_sigma], dim=-1)

        return output, featmap
        # return output[..., :2]
    
    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2 or 4]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3

        losses['reg_loss'] = self.loss(output[0], output[1], target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2 or 4]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        accuracy = dict()

        N = output[0].shape[0]
        output = output[0][..., :2]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['acc_pose'] = avg_acc

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if self.out_sigma:
            output[0][..., 2:] = output[0][..., 2:].sigmoid()

        if flip_pairs is not None:
            output_regression = fliplr_regression(
                output[0].detach().cpu().numpy(), flip_pairs)
        else:
            output_regression = output[0].detach().cpu().numpy()
        return output_regression

    def decode(self, img_metas, output, **kwargs):
        """Decode the keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, >=2]): predicted regression vector.
            kwargs: dict contains 'img_size'.
                img_size (tuple(img_width, img_height)): input image size.
        """
        batch_size = len(img_metas)
        sigma = output[..., 2:]
        output = output[..., :2]  # get prediction joint locations

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_regression(output, c, s,
                                                   kwargs['img_size'])
        if self.out_sigma:
            maxvals = (1 - sigma).mean(axis=2, keepdims=True)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)