import tensorflow as tf
import config as cfg
import keras.backend as K
import numpy as np

classes_num = cfg.CLASSES_NUM
anchors = np.asarray(cfg.ANCHORS)
ignore_thresh = cfg.THRESHOLD


def yolo_loss(args):
    """

    :param y_true: [batch_size, 13, 13, 125]
    :param y_pred: [batch_size, 13, 13, 125]
    :return:
    """
    y_true, y_pred = args[1], args[0]
    loss = 0
    y_true = K.reshape(y_true, shape=(-1, 13, 13, 5, 25))
    # 置信度
    object_mask = y_true[..., 4:5]
    noobject_mask = K.ones_like(object_mask, dtype=tf.float32) - object_mask
    # 类别
    true_class_probs = y_true[..., 5:]
    # anchors
    grid, raw_pred, pred_xy, pred_wh = cal_pred(y_pred, anchors)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # y_true是归一化到特征图上 减去偏移量得到 预测值
    raw_true_xy = y_true[..., :2] - grid
    # (batch_size, 13, 13, 5, 2) anchors (5, 2)
    raw_true_wh = K.log(y_true[..., 2:4] / anchors)
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
    # box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

    # calc iou, iterate over each of batch.
    ious = box_ious(K.reshape(y_true[..., 0:4], shape=(-1, 4)), K.reshape(pred_box, shape=(-1, 4)))
    # 边框损失
    # (batch_size, 13, 13, 5, 2)
    xy_loss = object_mask * K.square(raw_true_xy - raw_pred[..., 0:2])
    wh_loss = object_mask * K.square(raw_true_wh - raw_pred[..., 2:4])
    object_bbox_loss = cfg.COORD_OBJECT_SCALE * K.mean((K.sum(xy_loss, axis=[1, 2, 3, 4]) +
                                                        K.sum(wh_loss, axis=[1, 2, 3, 4])))

    # noobject_bbox_loss = noobject_mask * K.square()
    anchors_ = anchors.reshape((1, 1, 1, 5, 2))
    # 不负责预测目标的anchor的坐标损失。加了限制，希望直接回归到自身anchor box。
    # 中心点放到对一个cell的中心
    # noobject_xy_loss = noobject_mask * K.square(pred_xy - grid[..., 0:2] + 0.5)
    # noobject_wh_loss = noobject_mask * K.square(pred_wh - anchors_)
    # noobject_bbox_loss = cfg.CONF_NOOBJECT_SCALE * K.mean((K.sum(noobject_xy_loss, axis=[1, 2, 3, 4]) +
    #                                                       K.sum(noobject_wh_loss, axis=[1, 2, 3, 4])))

    # 类别损失
    cls_delta = object_mask * K.square(true_class_probs - raw_pred[..., 5:])
    cls_loss = cfg.CLASS_SCALE * K.mean(K.sum(cls_delta, axis=[1, 2, 3, 4]))

    # confidences 损失
    object_conf_delta = object_mask * K.square(raw_pred[..., 4:5] - ious)
    object_conf_loss = cfg.CONF_OBJECT_SCALE * K.mean(K.sum(object_conf_delta, axis=[1, 2, 3]))
    noobject_conf_delta = noobject_mask * K.square(raw_pred[..., 4:5])
    noobject_conf_loss = cfg.CONF_NOOBJECT_SCALE * K.mean(K.sum(noobject_conf_delta, axis=[1, 2, 3]))

    # loss += object_bbox_loss + noobject_bbox_loss + cls_loss + object_conf_loss + noobject_conf_loss
    loss += object_bbox_loss + cls_loss + object_conf_loss + noobject_conf_loss
    return loss


def box_ious(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(n, 4), xywh
    b2: tensor, shape=(n, 4), xywh

    Returns
    -------
    iou: tensor, shape=(n, 1)

    '''
    b1_wh = b1[:, 2:4]
    b2_wh = b2[:, 2:4]
    min_wh = K.minimum(b1_wh, b2_wh)
    intersect_area = min_wh[:, 0] * min_wh[:, 1]
    b1_area = b1_wh[:, 0] * b1_wh[:, 1]
    b2_area = b2_wh[:, 0] * b2_wh[:, 1]
    ious = intersect_area / (b1_area + b2_area - intersect_area)
    ious = K.reshape(ious, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.B, 1))
    return ious


def cal_pred(output, anchors):
    """

    :param output: (batch_size, 13, 13, 5 * (5 + num_class)
    :param anchors: (5, 2)
    :return:
    """
    # 5
    anchors_num = len(anchors)
    # (batch_size, 13, 13, 5, 2)
    anchors_tensor = K.reshape(K.constant(anchors, dtype=tf.float32), [1, 1, 1, anchors_num, 2])
    # height, width
    grid_shape = K.shape(output)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    # 没有指明axis 最后一个维度
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(output))
    # (batch_size, 13, 13, 5, (5 + num_class)
    output = K.reshape(
        output, [-1, grid_shape[0], grid_shape[1], anchors_num, classes_num + 5])
    # box_xy 得到在特征图上的大小
    box_xy = K.sigmoid(output[..., :2]) + grid
    # output: (batch_size, 13, 13, 5, 2) * anchors_tensor: (1, 1, 1, 5, 2)
    box_wh = K.exp(output[..., 2:4]) * anchors_tensor
    return grid, output, box_xy, box_wh


"""
损失函数公式见 loss.jpg
负责预测目标的anchor：

第一项：负责预测目标的anchor的坐标损失（包括中心定位和边界定位）。仅计算负责预测目标的那个anchor的坐标损失。（此项在yolo v1中边界定位是采用根号下的差值的平方）——衡量目标定位准确度
第三项：负责预测目标的anchor的confidence损失。负责预测物体的anchor需要计算confidence损失，confidence的目标就是让预测置信得分去逼近的预测bbox和Ground Truth的iou。——衡量可能有目标的准确度
第五项：负责预测目标的anchor的类别损失。每个类别的输出概率0-1之间，计算的是L2损失。也就是说分类问题也把它当做了回归问题。且yolo-V2中类别预测没有预测背景，因为置信得分值低于阈值就代表不存在目标。（此项与yolo-V1一致，只是因为不同anchor可同时预测不同目标了）——衡量目标分类的准确度
不负责预测目标的anchor：

第二项：不负责预测目标的anchor的坐标损失。加了限制，希望直接回归到自身anchor box。
第四项：不负责预测目标的anchor的confidece损失。（只计算那些与Groud Truth的IOU小于IOU阈值的anchor box）。首先，如果你与Groud Truth的IOU大于IOU阈值，且不负责预测目标，这种情况是很正常的，不予惩罚。但是，如果你的anchor box小于阈值说明本身就无意预测，那么我们干脆加上限制让你就回归到完全不预测的情况，即回归到与Ground Truth的IOU=0（此项在yolo-v1中没有）
"""
