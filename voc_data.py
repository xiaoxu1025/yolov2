import numpy as np
from exception import ValueValidException
import config as cfg
import cv2
from augment import DataAugment

# load config
cell_size = cfg.CELL_SIZE
im_size = cfg.IM_SIZE
w, h = cfg.IM_SIZE
img_channel_mean = cfg.IMG_CHANNEL_MEAN
img_scaling_factor = cfg.IMG_SCALING_FACTOR
classes_num = cfg.CLASSES_NUM
b = cfg.B


class VocData(object):
    def _parse_annotation(self, annotation):
        lines = annotation.strip().split()
        image_path = lines[0]
        gt_boxes = np.asarray([list(map(float, box.split(','))) for box in lines[1:]])
        return image_path, gt_boxes

    def data_generator_wrapper(self, annotations, batch_size):
        n = len(annotations)
        if n == 0 or batch_size <= 0:
            raise ValueValidException('样本数量为0或者batch_size小于等于0, please check it')
        return self._data_generator(annotations, batch_size)

    def _get_label(self, gt_boxes, anchors):
        label = np.zeros(shape=(cell_size, cell_size, b, 25))
        for idx in range(len(gt_boxes)):
            # x1, y1, x2, y2
            ctr_xy = (gt_boxes[idx, [0, 1]] + gt_boxes[idx, [2, 3]]) / 2
            # np.ceil() 向上取整 np.rint() 四舍五入 np.floor() 向下取整
            i, j = np.floor(ctr_xy).astype(np.int32)
            # 取出wh 这里是归一化的值
            box_wh = gt_boxes[idx, [2, 3]] - gt_boxes[idx, [0, 1]]
            # 计算iou 这里默认 将anchors的中心和gt_box的中心重合，所以只需要计算 最小的w 和 h即可
            min_wh = np.minimum(anchors, box_wh)
            # 相交area
            # (5, )
            intersect_area = min_wh[:, 0] * min_wh[:, 1]
            # 一个值
            box_area = box_wh[0] * box_wh[1]
            # (5, )
            anchors_area = anchors[:, 0] * anchors[:, 1]
            # (5, )
            iou = intersect_area / (anchors_area + box_area - intersect_area)
            # 最大的anchor
            argmax = np.argmax(iou)
            # class_index 从1开始
            class_index = gt_boxes[idx, 4].astype(np.int32)
            # x1, y1, x2, y2
            label[i, j, argmax, 0:4] = [ctr_xy[0], ctr_xy[1], box_wh[0], box_wh[1]]
            label[i, j, argmax, 4] = 1
            label[i, j, argmax, 4 + class_index] = 1
            # print(gt_boxes[idx])
            # print(label[i, j, argmax, :])
        return label

    def _data_generator(self, annotations, batch_size):
        i = 0
        n = len(annotations)
        data_augment = DataAugment()
        while True:
            img_data = []
            labels = []
            for _ in range(batch_size):
                if i == 0:
                    x = np.random.permutation(n)
                    annotations = annotations[x]
                annotation = annotations[i]
                image_path, gt_boxes = self._parse_annotation(annotation)
                # TODO 数据增广
                # data_augment(image_path, gt_boxes)
                # BGR -> RGB
                img = cv2.imread(image_path)
                # 原图宽高
                height, width = img.shape[:2]
                img = img[:, :, (2, 1, 0)]
                img = img.astype(np.float32)
                img[:, :, 0] -= img_channel_mean[0]
                img[:, :, 1] -= img_channel_mean[1]
                img[:, :, 2] -= img_channel_mean[2]
                img /= img_scaling_factor
                # resize
                img = cv2.resize(img, im_size, interpolation=cv2.INTER_CUBIC)
                img_data.append(img)
                # gt_boxes 归一化到特征图上
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] / width * cell_size
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] / height * cell_size
                # 得到聚类出来的anchors
                anchors = np.asarray(cfg.ANCHORS)
                label = self._get_label(gt_boxes, anchors)
                labels.append(label)
                i = (i + 1) % n
            img_data = np.asarray(img_data)
            labels = np.asarray(labels)
            yield [img_data, labels], np.zeros(batch_size)

#
# from voc_annotation import VOCAnnotation
#
# voc_annotation = VOCAnnotation('~/segment_data', 2007, 'train', './data/voc_classes.txt')
# annotations = voc_annotation.get_annotations()
#
# voc_data = VocData()
# voc_data_g = voc_data.data_generator_wrapper(annotations[:100], batch_size=10)
# img_data, labels = next(voc_data_g)
# print(img_data.shape, labels.shape)
