import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    """
    Args:
        model_cfg: AnchorHeadSingle的配置
        input_channels: 384 | 512 输入通道数
        num_class: 3
        class_names: ['Car','Pedestrian','Cyclist']
        grid_size: (X, Y, Z)
        point_cloud_range: (0, -39.68, -3, 69.12, 39.68, 1) ，[0, -40, -3, 70.4, 40, 1]
        predict_boxes_when_training: False
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        # 每个点都有3个尺度的个先验框  每个先验框都有两个方向（0度，90度） num_anchors_per_location:[2, 2, 2]
        # 每个类别有两个anchor然后这里计算每个地方一共多少个anchor
        self.num_anchors_per_location = sum(self.num_anchors_per_location) # sum([2, 2, 2])

        # Conv2d(512,18,kernel_size=(1,1),stride=(1,1))
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        # Conv2d(512,42,kernel_size=(1,1),stride=(1,1))
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        # 如果存在方向损失，则添加方向卷积层Conv2d(512,12,kernel_size=(1,1),stride=(1,1))
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    # 初始化参数
    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi)) # 初始化分类卷积偏置
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001) # 初始化分类卷积权重

    def forward(self, data_dict):
        # 从字典中取出经过backbone处理过的信息  spatial_features_2d 维度 （batch_size, W, H, 176）   （2,512,200,176）,后面两个维度才能参与2d卷积
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d) # 每个坐标点上面6个先验框的类别预测 --> (batch_size, 18, W, H)
        box_preds = self.conv_box(spatial_features_2d) # 每个坐标点上面6个先验框的参数预测 --> (batch_size, 42, W, H)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]  
        # 维度调整，将先验框调整参数放置在最后一维度   [N, H, W, C] --> (batch_size ,W, H, 42)

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        # 如果是在训练模式的时候，需要对每个先验框分配GT来计算loss
        if self.training:
            # targets_dict = {
            #     'box_cls_labels': cls_labels, # (4，211200）
            #     'box_reg_targets': bbox_targets, # (4，211200, 7）
            #     'reg_weights': reg_weights # (4，211200）
            # }
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            # 将GT分配结果放入前向传播字典中
            self.forward_ret_dict.update(targets_dict)

        # 如果不是训练模式，则直接生成进行box的预测，在PV-RCNN中在训练时候也要生成bbox用于refinement
        if not self.training or self.predict_boxes_when_training:
            # 根据预测结果解码生成最终结果
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
