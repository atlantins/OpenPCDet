import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils

'''
    POINT_HEAD:
        NAME: PointHeadSimple
        CLS_FC: [256, 256]
        CLASS_AGNOSTIC: True
        USE_POINT_FEATURES_BEFORE_FUSION: True
        TARGET_CONFIG:
            GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
        LOSS_CONFIG:
            LOSS_REG: smooth-l1
            LOSS_WEIGHTS: {
                'point_cls_weight': 1.0,
            }

'''

class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss

    # CLS_FC: [256, 256] 
    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        # 得到一批数据中batch_size的大小，以方便逐帧完成target assign
        batch_size = gt_boxes.shape[0]
        # 得到一批数据中所有点云的batch_id
        bs_idx = points[:, 0]
        # 初始化每个点云的类别，默认全0属于背景； shape （batch * 16384）
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        # 初始化每个点云预测box的参数，默认全0； shape （batch * 16384, 8）
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        # None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        # 逐帧点云数据进行处理
        for k in range(batch_size):
            # 得到一个mask，用于取出一批数据中属于当前帧的点
            bs_mask = (bs_idx == k)
            # 取出对应的点shape (16384, 3), PV-RCNN关键点(2048, 3)
            points_single = points[bs_mask][:, 1:4]
            # 初始化当前帧中点的类别，默认为0背景， (16384, ), PV-RCNN关键点(2048,)
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            '''
            PV-RCNN中点的数量是2048或者4096
            points_single : (16384, 3) --> (1, 16384, 3)
            gt_boxes : (batch, num_of_GTs, 8) --> (当前帧的GT, num_of_GTs, 8)
            box_idxs_of_pts : (16384, )，其中点云分割中背景为-1, 前景点指向GT中的索引，
            例如[-1,-1,3,20,-1,0]，其中，3,20,0分别指向第3个、第20个和第0个GT
            '''
            # 计算哪些中点在GTbox, box_idxs_of_pts
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            # mask 表明该帧中的哪些点属于前景点，哪些点属于背景点;得到属于前景点的mask
            box_fg_flag = (box_idxs_of_pts >= 0)
            # 是否忽略在enlarge box中的点 True
            if set_ignore_flag:
                # 计算哪些点在GTbox_enlarge中
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                # GTBox内的点
                fg_flag = box_fg_flag
                # ^为异或运算符，不同为真，相同为假，这样就可以得到哪些点在GT enlarge中了
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                # 将这些真实GT边上的点设置为-1      loss计算时，不考虑这类点
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            # [box_idxs_of_pts[fg_flag]]取出所有点中属于前景的点，
            # 并为这些点分配对应的GT_box shape (num_of_gt_match_by_points, 8)
            # 8个维度分别是x, y, z, l, w, h, heading, class_id
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            # 将类别信息赋值给对应的前景点 (16384, )
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            # 赋值点的类别GT结果到的batch中对应的帧位置
            point_cls_labels[bs_mask] = point_cls_labels_single
            # 如果该帧中GT的前景点的数量大于0且需要预测每个点的box PV-RCNN中不需要，PointRCNN中需要
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                # 初始化该帧中box的8个回归参数，并置0
                # 此处编码为(Δx, Δy, Δz, dx, dy, dz, cos(heading), sin(heading)) 8个
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                # 对属于前景点的box进行编码 得到的是 （num_of_fg_points, 8）
                # 其中8是(Δx, Δy, Δz, dx, dy, dz, cos(heading), sin(heading))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                # 将每个前景点的box信息赋值到该帧中box参数预测中
                # fg_point_box_labels: (num_of_GT_matched_by_point,8)
                # point_box_labels_single: (16384, 8)
                point_box_labels_single[fg_flag] = fg_point_box_labels
                # 赋值点的回归编码结果到的batch中对应的帧位置
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        # 将每个点的类别、每个点对应box的7个回归参数放入字典中
        targets_dict = {
            # 将一个batch中所有点的GT类别结果放入字典中 shape (batch * 16384)
            'point_cls_labels': point_cls_labels,
            # 将一个batch中所有点的GT_box编码结果放入字典中 shape (batch * 16384) shape (batch * 16384, 8)
            'point_box_labels': point_box_labels,
            # PVRCNN NONE
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1) # 第一阶段点的GT类别
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class) # 第一阶段点的预测类别

        '''
        point_cls_labels[0:10] tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], device='cuda:0')
        positives[0:10] tensor([False, False, False, False, False, False,  True, False, False, False],device='cuda:0')
        negative_cls_weights[0:10]  tensor([1., 1., 1., 1., 1., 1., 0., 1., 1., 1.], device='cuda:0')
        cls_weights[0:10] tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
        pos_normalizer 637
        cls_weights tensor([0.0016, 0.0016, 0.0016, 0.0016, 0.0016, 0.0016, 0.0016, 0.0016, 0.0016,
        相当于多少个正样本,平均他们的wegiht
        '''
        positives = (point_cls_labels > 0)  # 取出属于前景的点的mask，0为背景，1,2,3分别为前景，-1不关注
        negative_cls_weights = (point_cls_labels == 0) * 1.0 # 背景点分类权重置0,本来就是0乘了之后也是0
        cls_weights = (negative_cls_weights + 1.0 * positives).float() # 前景点分类权重置0
        pos_normalizer = positives.sum(dim=0).float() # 使用前景点的个数来normalize，使得一批数据中每个前景点贡献的loss一样
        cls_weights /= torch.clamp(pos_normalizer, min=1.0) # 正则化每个类别分类损失权重

        # 初始化分类的one-hot （batch * 16384, 4）      point_cls_labels.shape [4096]
        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1) 
        # 将目标标签转换为one-hot编码形式   .long将true转成1，false转成0 
        # scatter 就是将b里面按照index，也就是上面的.long放置，这里面其实只有前景，最后生成了独热向量，也就是在哪个类别，哪个类别就是1
        # https://blog.csdn.net/guofei_fly/article/details/104308528
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:] # 假设了第一维为背景维
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights) # 计算分类损失使用focal loss
        point_loss_cls = cls_loss_src.sum()
 
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS # 分类损失权重
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight'] # 分类损失乘以分类损失权重
        if tb_dict is None:
            tb_dict = {}
        # 使用.item（）将tensor转换成标量，抛弃Backward属性，可以优化显存，
        tb_dict.update({  
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)

        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
