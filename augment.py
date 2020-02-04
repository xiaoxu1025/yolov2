import cv2
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


class Rotate(object):
    # 水平垂直翻转
    FLIP = -1
    # 垂直翻转
    VERTICAL_FLIP = 0
    # 水平翻转
    HORIZONTAL_FLIP = 1
    # 旋转90
    ROTATE_90 = 2
    # 旋转180
    ROTATE_180 = 3
    # 旋转270
    ROTATE_270 = 4

    def __init__(self, *args, **kwargs):
        # 这里把参数都传过来了 虽然这里用不到 可以不传
        self.__dict__.update(kwargs)

    def rotate(self, angle):
        if angle == self.ROTATE_90:
            img, gt_boxes = self.rotate_90()
        elif angle == self.ROTATE_180:
            img, gt_boxes = self.rotate_180()
        elif angle == self.ROTATE_270:
            img, gt_boxes = self.rotate_270()
        elif angle == self.FLIP:
            img, gt_boxes = self.flip()
        elif angle == self.HORIZONTAL_FLIP:
            img, gt_boxes = self.horizontal_flip()
        elif angle == self.VERTICAL_FLIP:
            img, gt_boxes = self.vertical_flip()
        else:
            raise ValueError('angle: %s not support only support ROTATE_90, ROTATE_180, ROTATE_270' % angle)
        return img, gt_boxes

    def rotate_90(self):
        # height = self.height
        width = self.width
        img = self.img
        gt_boxes = self.gt_boxes
        flip = np.random.rand()
        if flip < self.rand:
            # 逆时针旋转
            if self.cv2:
                img = cv2.flip(img, 1)
                img = np.transpose(img, (1, 0, 2))
            else:
                img = img.transpose(Image.ROTATE_90)
            for idx in range(len(gt_boxes)):
                gt_box = gt_boxes[idx]
                x1, y1, x2, y2 = gt_box[0:4]
                gt_box[0] = y1
                gt_box[1] = width - x2
                gt_box[2] = y2
                gt_box[3] = width - x1
        return img, gt_boxes

    def rotate_180(self):
        height = self.height
        width = self.width
        img = self.img
        gt_boxes = self.gt_boxes
        flip = np.random.rand()
        if flip < self.rand:
            if self.cv2:
                img = cv2.flip(img, -1)
            else:
                img = img.transpose(Image.ROTATE_180)
            for idx in range(len(gt_boxes)):
                gt_box = gt_boxes[idx]
                x1, y1, x2, y2 = gt_box[0:4]
                gt_box[0] = width - x2
                gt_box[1] = height - y2
                gt_box[2] = width - x1
                gt_box[3] = height - y1
        return img, gt_boxes

    def rotate_270(self):
        height = self.height
        # width = self.width
        img = self.img
        gt_boxes = self.gt_boxes
        flip = np.random.rand()
        if flip < self.rand:
            # 逆时针旋转
            if self.cv2:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 0)
                img = cv2.flip(img, -1)
            else:
                img = img.transpose(Image.ROTATE_270)
            for idx in range(len(gt_boxes)):
                gt_box = gt_boxes[idx]
                x1, y1, x2, y2 = gt_box[0:4]
                gt_box[0] = height - y2
                gt_box[1] = x1
                gt_box[2] = height - y1
                gt_box[3] = x2
        return img, gt_boxes

    def horizontal_flip(self):
        width = self.width
        img = self.img
        gt_boxes = self.gt_boxes
        flip = np.random.rand()
        if flip < self.rand:
            # 水平翻转
            if self.cv2:
                img = cv2.flip(img, 1)
            else:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = width - gt_boxes[:, [2, 0]]
        return img, gt_boxes

    def vertical_flip(self):
        height = self.height
        img = self.img
        gt_boxes = self.gt_boxes
        flip = np.random.rand()
        if flip < self.rand:
            # 垂直翻转
            if self.cv2:
                img = cv2.flip(img, 0)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            gt_boxes[:, [1, 3]] = height - gt_boxes[:, [3, 1]]
        return img, gt_boxes

    def flip(self):
        """
        水平垂直翻转
        :return:
        """
        height = self.height
        width = self.width
        img = self.img
        gt_boxes = self.gt_boxes
        flip = np.random.rand()
        if flip < self.rand:
            # 水平加垂直翻转
            if self.cv2:
                img = cv2.flip(img, -1)
            else:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            gt_boxes[:, [0, 2]] = width - gt_boxes[:, [2, 0]]
            gt_boxes[:, [1, 3]] = height - gt_boxes[:, [3, 1]]
        return img, gt_boxes


class DataAugment(Rotate):

    def __init__(self, use_cv2=True, rand=.5, **kwargs):
        """
        是否采用cv2处理
        :param use_cv2: True  cv2 fase PIL Image
        :param rand: 水平垂直翻转概率
        """
        self.cv2 = use_cv2
        self.rand = rand
        super(DataAugment, self).__init__(use_cv2, rand, **kwargs)

    def __call__(self, image_path, gt_boxes, **kwargs):
        if isinstance(gt_boxes, list):
            gt_boxes = np.array(gt_boxes)
        self.gt_boxes = gt_boxes
        self.img, self.width, self.height = self._parse_image(image_path)
        return self

    def _parse_image(self, image_path):
        if self.cv2:
            # BGR像素存储，如果转换成RGB则需要用cvtColor函数进行转换
            # image_arr_rgb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
            # 或者 img = img[:, :, (2, 1, 0)]
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
        else:
            img = Image.open(image_path)
            width, height = img.size
        return img, width, height

    def add_noise(self, image, hue=.1, sat=1.5, val=1.5):
        # 给图片添加噪声
        image_max = np.max(image)
        image = image if image_max <= 1. else image / 255.
        hue = self._rand(-hue, hue)
        sat = self._rand(1, sat) if self._rand() < .5 else 1 / self._rand(1, sat)
        val = self._rand(1, val) if self._rand() < .5 else 1 / self._rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)
        return image_data

    def _rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
