import tensorflow as tf


def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
         max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    inter = 0 if tb < 0 or lr < 0 else tb * lr
    return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


def cal_iou(bboxes1, bboxes2):
    """

    :param bboxes1: (batch_size, 7, 7, 2, 4)
    :param bboxes2: (batch_size, 7, 7, 1, 4)
    :return:
    """
    # 将中心点坐标转换成 x1, y1, x2, y2
    _bboxes1 = tf.stack([
        bboxes1[..., 0] - (bboxes1[..., 2] - 1) * 0.5,
        bboxes1[..., 1] - (bboxes1[..., 3] - 1) * 0.5,
        bboxes1[..., 0] + (bboxes1[..., 2] - 1) * 0.5,
        bboxes1[..., 1] + (bboxes1[..., 3] - 1) * 0.5
    ], axis=-1)

    _bboxes2 = tf.stack([
        bboxes2[..., 0] - (bboxes2[..., 2] - 1) * 0.5,
        bboxes2[..., 1] - (bboxes2[..., 3] - 1) * 0.5,
        bboxes2[..., 0] + (bboxes2[..., 2] - 1) * 0.5,
        bboxes2[..., 1] + (bboxes2[..., 3] - 1) * 0.5
    ], axis=-1)

    iw = tf.maximum(
        tf.minimum(_bboxes1[..., 2], _bboxes2[..., 2]) - tf.maximum(_bboxes1[..., 0], _bboxes2[..., 0]) + 1, 0)
    ih = tf.maximum(
        tf.minimum(_bboxes1[..., 3], _bboxes2[..., 3]) - tf.maximum(_bboxes1[..., 1], _bboxes2[..., 1]) + 1, 0)
    inter = iw * ih
    square1 = _bboxes1[..., 2] * _bboxes1[..., 3]
    square2 = _bboxes2[..., 2] * _bboxes2[..., 3]
    union_square = tf.maximum(square1 + square2 - inter, 1e-10)
    return tf.clip_by_value(inter / union_square, 0.0, 1.0)
