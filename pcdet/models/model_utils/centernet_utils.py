# This file is modified from https://github.com/tianweiy/CenterPoint

import torch
import torch.nn.functional as F
import numpy as np
import numba


def gaussian_radius(height, width, min_overlap=0.5):
    """
    计算高斯分布的半径
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    # 预测框两个角点在GT框的两个角点以r为半径的圆内，如何确定半径r，保证预测框与真值框的IOU大于一个阈值
    """
    1.一角点在真值框内,一角点在真值框外
    最小IOU在预测框两个角点分别和和半径r的圆相外切和相内切时取得(例如可以固定某一角点在x方向不变,变动y方向观察相交、相并面积的变化情况)
    因此我们只需要考虑“预测的框和GTbox两个角点以r为半径的圆一个边内切,一个边外切
    min_overlap =(h-r)*(w-r)/(2*h*w-(h-r)*(w-r)) --> r
    整理为r的一元二次方程: r^2 - (h+w)*r + (1-min_overlap)*h*w / (1+min_overlap) =0
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    # 计算高斯分布的边界
    m, n = [(ss - 1.) / 2. for ss in shape]  # 计算中心点坐标
    y, x = np.ogrid[-m:m + 1, -n:n + 1]  # 生成二维坐标系，并表示y和x方向的坐标范围

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))  # e^-(x^2+y^2)/2*&^2   只与点到中心点的位置有关
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # np.finfo(h.dtype).eps表示h数组中最小的非负数，  finfo获得浮点类型的精度信息
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    # 在heatmap上画高斯分布，每一个类别一个heatmap，每一个点不是叠加，而是一直不断选最大值
    diameter = 2 * radius + 1  # 因为半径为整数，直径为奇数时，中心点坐标可以被确定。
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # 将高斯分布结果约束在边界内
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()
        # 将高斯分布覆盖到heartmap上，相当于不断的在heartmap基础上添加关键点的高斯分布
        # 即同一种类型的框会在一个heartmap 某一类类别通道上面不断添加
        # 最终通过函数总体的for循环，相当于不断将目标画在heartmap
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep


def _gather_feat(feat, ind, mask=None):
    # 根据inds取feat
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    # 将特征维度变换，使用_gather_feat就是根据索引得到feat
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    # 返回heatmap里面最大的K个score，索引，类别，y坐标，x坐标
    batch, num_class, height, width = scores.size() # 输入heatmap (4,3,200,176)

    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K) # 选取前500个

    topk_inds = topk_inds % (height * width)  # (4,3,500)
    topk_ys = (topk_inds // width).float()    # (4,3,500)
    topk_xs = (topk_inds % width).int().float() # (4,3,500)

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) # 将多类的得分合并,选其中的前500
    topk_classes = (topk_ind // K).int() # 获得前k个最大得分的类别
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K) #
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                             circle_nms=False, score_thresh=None, post_center_limit_range=None):
    batch_size, num_class, _, _ = heatmap.size()
    '''
    这段代码实现了从检测头（heatmap, rot_cos, rot_sin, center, center_z, dim）中解码出物体的边界框（final_box_preds）
    以及预测的类别得分（final_scores）和类别标签（final_class_ids），
    再映射回原来的点云空间
    将boxes、scores、labels保存到原来的ret_pred_dicts中
    '''

    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)

    scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)
    center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
    rot_sin = _transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
    rot_cos = _transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
    center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
    ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]
    if vel is not None:
        vel = _transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
        box_part_list.append(vel)

    final_box_preds = torch.cat((box_part_list), dim=-1)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)  # .all(2)表示与操作，也就是两个都要大于最小值
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2) # 两个都要小于最大值

    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]

        if circle_nms:
            assert False, 'not checked yet'
            centers = cur_boxes[:, [0, 1]]
            boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
            keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)

            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })
    return ret_pred_dicts
